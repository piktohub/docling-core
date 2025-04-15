#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Chunker implementation leveraging the document structure."""

from __future__ import annotations

import logging
import re
from typing import Any, ClassVar, Final, Iterator, Literal, Optional

from pydantic import ConfigDict, Field, StringConstraints, field_validator
from typing_extensions import Annotated, override

from docling_core.experimental.serializer.base import (
    BaseDocSerializer,
    BaseSerializerProvider,
    BaseTableSerializer,
    SerializationResult,
)
from docling_core.experimental.serializer.common import create_ser_result
from docling_core.experimental.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
)
from docling_core.search.package import VERSION_PATTERN
from docling_core.transforms.chunker import BaseChunk, BaseChunker, BaseMeta
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.base import ImageRefMode
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import (
    DocItem,
    DoclingDocument,
    DocumentOrigin,
    InlineGroup,
    LevelNumber,
    OrderedList,
    SectionHeaderItem,
    TableItem,
    TitleItem,
    UnorderedList,
    TextItem,
    PictureItem,
    ListItem,
)

_VERSION: Final = "1.0.0"

_KEY_SCHEMA_NAME = "schema_name"
_KEY_VERSION = "version"
_KEY_DOC_ITEMS = "doc_items"
_KEY_HEADINGS = "headings"
_KEY_CAPTIONS = "captions"
_KEY_ORIGIN = "origin"

_logger = logging.getLogger(__name__)


class DocMeta(BaseMeta):
    """Data model for Hierarchical Chunker chunk metadata."""

    schema_name: Literal["docling_core.transforms.chunker.DocMeta"] = Field(
        default="docling_core.transforms.chunker.DocMeta",
        alias=_KEY_SCHEMA_NAME,
    )
    version: Annotated[str, StringConstraints(pattern=VERSION_PATTERN, strict=True)] = (
        Field(
            default=_VERSION,
            alias=_KEY_VERSION,
        )
    )
    doc_items: list[DocItem] = Field(
        alias=_KEY_DOC_ITEMS,
        min_length=1,
    )
    headings: Optional[list[str]] = Field(
        default=None,
        alias=_KEY_HEADINGS,
        min_length=1,
    )
    captions: Optional[list[str]] = Field(  # deprecated
        deprecated=True,
        default=None,
        alias=_KEY_CAPTIONS,
        min_length=1,
    )
    origin: Optional[DocumentOrigin] = Field(
        default=None,
        alias=_KEY_ORIGIN,
    )

    excluded_embed: ClassVar[list[str]] = [
        _KEY_SCHEMA_NAME,
        _KEY_VERSION,
        _KEY_DOC_ITEMS,
        _KEY_ORIGIN,
    ]
    excluded_llm: ClassVar[list[str]] = [
        _KEY_SCHEMA_NAME,
        _KEY_VERSION,
        _KEY_DOC_ITEMS,
        _KEY_ORIGIN,
    ]

    @field_validator(_KEY_VERSION)
    @classmethod
    def check_version_is_compatible(cls, v: str) -> str:
        """Check if this meta item version is compatible with current version."""
        current_match = re.match(VERSION_PATTERN, _VERSION)
        doc_match = re.match(VERSION_PATTERN, v)
        if (
            doc_match is None
            or current_match is None
            or doc_match["major"] != current_match["major"]
            or doc_match["minor"] > current_match["minor"]
        ):
            raise ValueError(f"incompatible version {v} with schema version {_VERSION}")
        else:
            return _VERSION


class DocChunk(BaseChunk):
    """Data model for document chunks."""

    meta: DocMeta


class TripletTableSerializer(BaseTableSerializer):
    """Triplet-based table item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TableItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        parts: list[SerializationResult] = []

        cap_res = doc_serializer.serialize_captions(
            item=item,
            **kwargs,
        )
        if cap_res.text:
            parts.append(cap_res)

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            table_df = item.export_to_dataframe()
            if table_df.shape[0] >= 1 and table_df.shape[1] >= 2:

                # copy header as first row and shift all rows by one
                table_df.loc[-1] = table_df.columns  # type: ignore[call-overload]
                table_df.index = table_df.index + 1
                table_df = table_df.sort_index()

                rows = [str(item).strip() for item in table_df.iloc[:, 0].to_list()]
                cols = [str(item).strip() for item in table_df.iloc[0, :].to_list()]

                nrows = table_df.shape[0]
                ncols = table_df.shape[1]
                table_text_parts = [
                    f"{rows[i]}, {cols[j]} = {str(table_df.iloc[i, j]).strip()}"
                    for i in range(1, nrows)
                    for j in range(1, ncols)
                ]
                table_text = ". ".join(table_text_parts)
                parts.append(create_ser_result(text=table_text, span_source=item))

        text_res = "\n\n".join([r.text for r in parts])

        return create_ser_result(text=text_res, span_source=parts)


class ChunkingDocSerializer(MarkdownDocSerializer):
    """Doc serializer used for chunking purposes."""

    table_serializer: BaseTableSerializer = TripletTableSerializer()
    params: MarkdownParams = MarkdownParams(
        image_mode=ImageRefMode.PLACEHOLDER,
        image_placeholder="",
        escape_underscores=False,
        escape_html=False,
    )


class ChunkingSerializerProvider(BaseSerializerProvider):
    """Serializer provider used for chunking purposes."""

    @override
    def get_serializer(self, doc: DoclingDocument) -> BaseDocSerializer:
        """Get the associated serializer."""
        return ChunkingDocSerializer(doc=doc)


class HierarchicalChunker(BaseChunker):
    r"""Chunker implementation leveraging the document layout.

    Args:
        merge_list_items (bool): Whether to merge successive list items.
            Defaults to True.
        delim (str): Delimiter to use for merging text. Defaults to "\n".
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    serializer_provider: BaseSerializerProvider = ChunkingSerializerProvider()

    # deprecated:
    merge_list_items: Annotated[bool, Field(deprecated=True)] = True

    def chunk(
        self,
        dl_doc: DLDocument,
        **kwargs: Any,
    ) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.

        Args:
            dl_doc (DLDocument): document to chunk

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        my_doc_ser = self.serializer_provider.get_serializer(doc=dl_doc)
        heading_by_level: dict[LevelNumber, str] = {}
        prev_heading_by_level: dict[LevelNumber, str] = {}
        visited: set[str] = set()
        ser_res = create_ser_result()
        excluded_refs = my_doc_ser.get_excluded_refs(**kwargs)
        duplicated_picture_items: list[PictureItem] = []
        duplicated_section_header_items: list[SectionHeaderItem] = []
        duplicated_text_items: list[TextItem] = []
        for item in self._get_duplicated_items(dl_doc):
            if isinstance(item, PictureItem):
                duplicated_picture_items.append(item)
            elif isinstance(item, SectionHeaderItem):
                duplicated_section_header_items.append(item)
            elif isinstance(item, TextItem):
                duplicated_text_items.append(item)
         # To detect if picture item existed before skipping. Only once per same image
        picture_items: list[PictureItem] = []
        for item, level in dl_doc.iterate_items(with_groups=True):
            item = self._cleaned_item(item)
            if item.self_ref in excluded_refs:
                continue
            if isinstance(item, (TitleItem, SectionHeaderItem)):
                level = item.level if isinstance(item, SectionHeaderItem) else 0
                heading_by_level[level] = item.text

                # remove headings of higher level as they just went out of scope
                keys_to_del = [k for k in heading_by_level if k > level]
                for k in keys_to_del:
                    heading_by_level.pop(k, None)
                continue
            elif (
                isinstance(item, (OrderedList, UnorderedList, InlineGroup, DocItem))
                and item.self_ref not in visited
            ):
                ser_res = my_doc_ser.serialize(item=item, visited=visited)
            else:
                continue

            if not ser_res.text:
                continue
            if doc_items := [u.item for u in ser_res.spans]:
                c = DocChunk(
                    text=ser_res.text,
                    meta=DocMeta(
                        doc_items=doc_items,
                        headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                        or None,
                        origin=dl_doc.origin,
                    ),
                )
                yield c

    def _is_bbox_match(self, prov_item, current_prov_item):
        return (
            int(prov_item.bbox.l) == int(current_prov_item.bbox.l)
        ) and (
            int(prov_item.bbox.t) == int(current_prov_item.bbox.t)
        ) and (
            int(prov_item.bbox.r) == int(current_prov_item.bbox.r)
        ) and (
            int(prov_item.bbox.b) == int(current_prov_item.bbox.b)
        )

    def _is_same_location(self, current_item, item):
        if item.prov == [] or current_item.prov == []:
            return False

        item_prov = item.prov[0]
        current_item_prov = current_item.prov[0]

        return self._is_bbox_match(item_prov, current_item_prov)

    def _picture_item_existed(self, current_picture_item: PictureItem, picture_items: list[PictureItem]):
        if current_picture_item.image is None: return False
        # Here, we will compare ImageData, and return true if found.
        # But when imageData is blank, we will compare the:
        # 1. BoundingBox (Ideally they are the same when a page is duplicated)
        picture_item_found = False
        for picture_item in picture_items:
            if picture_item.image is None:
                # When picture item is not rendered, exit immediately.
                picture_item_found = True
                break
            else:
                if picture_item.image.uri == current_picture_item.image.uri:
                    picture_item_found = True
                    break

        return picture_item_found

    def _section_header_item_existed(self, current_section_header_item: SectionHeaderItem, section_header_items: list[SectionHeaderItem]):
        section_header_item_found = False

        for section_header_item in section_header_items:
            if section_header_item.text == current_section_header_item.text and self._is_same_location(current_section_header_item, section_header_item):
                section_header_item_found = True
                break

        return section_header_item_found

    def _text_item_existed(self, current_text_item: TextItem, text_items: list[TextItem]):
        text_item_found = False

        for text_item in text_items:
            if text_item.text == current_text_item.text and self._is_same_location(current_text_item, text_item):
                text_item_found = True
                break

        return text_item_found

    def _get_duplicated_items(self, dl_doc):
        picture_items: list[PictureItem] = []
        section_header_items: list[SectionHeaderItem] = []
        text_items: list[TextItem] = []
        for item, level in dl_doc.iterate_items():
            if isinstance(item, PictureItem):
                if self._picture_item_existed(current_picture_item=item, picture_items=picture_items):
                    yield item
                else:
                    picture_items.append(item)

            if isinstance(item, SectionHeaderItem):
                if self._section_header_item_existed(current_section_header_item=item, section_header_items=section_header_items):
                    yield item
                else:
                    section_header_items.append(item)

            if isinstance(item, TextItem):
                if self._text_item_existed(current_text_item=item, text_items=text_items):
                    yield item
                else:
                    text_items.append(item)

    def _cleanup_list(self, text):
        return text.replace("● ", "").replace(" ", "")

    def _cleanup_text(self, text):
        return text.replace(u"\u200b", u"").replace(u"\t", u"").replace("\xa0", " ")

    def _cleaned_item(self, item):
        if not hasattr(item, 'text'): return item

        if isinstance(
            item, ListItem
        ) or (  # TODO remove when all captured as ListItem:
            isinstance(item, TextItem)
            and item.label == DocItemLabel.LIST_ITEM
        ):
            item.text = self._cleanup_list(item.text)
        else:
            item.text = self._cleanup_text(item.text)

        return item