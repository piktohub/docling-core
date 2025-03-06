#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

"""Chunker implementation leveraging the document structure."""

from __future__ import annotations

import io
import logging
import re
from typing import Any, ClassVar, Final, Iterator, Literal, Optional, Union

from pandas import DataFrame
from pydantic import Field, StringConstraints, field_validator
from typing_extensions import Annotated

from docling_core.search.package import VERSION_PATTERN
from docling_core.transforms.chunker import BaseChunk, BaseChunker, BaseMeta
from docling_core.types import DoclingDocument as DLDocument
from docling_core.types.doc.document import (
    CodeItem,
    DocItem,
    DocumentOrigin,
    LevelNumber,
    ListItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    PictureItem,
)
from docling_core.types.doc.labels import DocItemLabel

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
    captions: Optional[list[str]] = Field(
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


class HierarchicalChunker(BaseChunker):
    r"""Chunker implementation leveraging the document layout.

    Args:
        merge_list_items (bool): Whether to merge successive list items.
            Defaults to True.
        delim (str): Delimiter to use for merging text. Defaults to "\n".
    """

    merge_list_items: bool = True

    @classmethod
    def _triplet_serialize(cls, table_df: DataFrame) -> str:
        in_memory = io.BytesIO()
        table_df.to_csv(in_memory)
        in_memory.seek(0)

        return in_memory.read()

    def chunk(self, dl_doc: DLDocument, **kwargs: Any) -> Iterator[BaseChunk]:
        r"""Chunk the provided document.

        Args:
            dl_doc (DLDocument): document to chunk

        Yields:
            Iterator[Chunk]: iterator over extracted chunks
        """
        prev_heading_by_level: dict[LevelNumber, str] = {}
        heading_by_level: dict[LevelNumber, str] = {}
        list_items: list[TextItem] = []
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

        for item, level in dl_doc.iterate_items():
            captions = None
            if isinstance(item, DocItem):
                item = self._cleaned_item(item)

                if hasattr(item, 'text') and not self._have_text(item):
                    # print(f"CHChunker: Skipped None Text: {item.text[0:20]}")
                    continue

                if isinstance(item, SectionHeaderItem):
                    if self._section_header_item_existed(current_section_header_item=item, section_header_items=duplicated_section_header_items):
                        # print(f"CHChunker: Skipped Header: {item.text[0:20]}")
                        continue

                if isinstance(item, TextItem) and not item.label in (DocItemLabel.TITLE, DocItemLabel.LIST_ITEM):
                    if self._text_item_existed(current_text_item=item, text_items=duplicated_text_items):
                        # print(f"CHChunker: Skipped Text: {item.text[0:20]}")
                        continue

                if item.label in [DocItemLabel.PAGE_FOOTER]:
                    # print(f"CHChunker: Skipped Footer: {item.text[0:20]}")
                    continue

                # first handle any merging needed
                if self.merge_list_items:
                    if isinstance(
                        item, ListItem
                    ) or (  # TODO remove when all captured as ListItem:
                        isinstance(item, TextItem)
                        and item.label == DocItemLabel.LIST_ITEM
                    ):
                        # print(f"CHChunker: Add List Item {item.ilevel} {item.text[0:20]}")
                        list_items.append(item)
                        continue
                    elif list_items:  # need to yield
                        # Reset
                        prev_heading_by_level = {}

                        yield DocChunk(
                            text=self.delim.join([i.text for i in list_items]),
                            meta=DocMeta(
                                doc_items=list_items,
                                headings=[
                                    heading_by_level[k]
                                    for k in sorted(heading_by_level)
                                ]
                                or None,
                                origin=dl_doc.origin,
                            ),
                        )
                        list_items = []  # reset

                if isinstance(item, SectionHeaderItem) or (
                    isinstance(item, TextItem)
                    and item.label in [DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE]
                ):
                    level = (
                        item.level
                        if isinstance(item, SectionHeaderItem)
                        else (0 if item.label == DocItemLabel.TITLE else 1)
                    )
                    doc_chunk = self._get_doc_chunk_heading(heading_by_level, prev_heading_by_level, item, captions, dl_doc)
                    if doc_chunk is not None:
                        yield doc_chunk
                        prev_heading_by_level = {}

                    prev_heading_by_level = self._get_prev_heading_by_level(item, heading_by_level, prev_heading_by_level, level)

                    heading_by_level[level] = item.text

                    # remove headings of higher level as they just went out of scope
                    keys_to_del = [k for k in heading_by_level if k > level]
                    for k in keys_to_del:
                        heading_by_level.pop(k, None)
                    continue

                if (
                    isinstance(item, TextItem)
                    or ((not self.merge_list_items) and isinstance(item, ListItem))
                    or isinstance(item, CodeItem)
                    and (not item.label == DocItemLabel.PAGE_FOOTER)
                ):
                    text = item.text
                elif isinstance(item, TableItem):
                    table_df = item.export_to_dataframe()
                    if table_df.shape[0] < 1 or table_df.shape[1] < 2:
                        # at least two cols needed, as first column contains row headers
                        continue
                    text = self._triplet_serialize(table_df=table_df)
                    captions = [
                        c.text for c in [r.resolve(dl_doc) for r in item.captions]
                    ] or None
                elif isinstance(item, PictureItem):
                    if self._picture_item_existed(current_picture_item=item, picture_items=picture_items): continue

                    picture_items.append(item)
                    text = "." # Meant to set dot so that it is not removed due to empty for PICTURE
                    captions = [
                        c.text for c in [r.resolve(dl_doc) for r in item.captions]
                    ] or None
                else:
                    continue
                # Reset
                prev_heading_by_level = {}
                # print(f"CHChunker: Text {text[0:20]}")
                c = DocChunk(
                    text=text,
                    meta=DocMeta(
                        doc_items=[item],
                        headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                        or None,
                        captions=captions,
                        origin=dl_doc.origin,
                    ),
                )
                yield c

        # Reset
        prev_heading_by_level = {}
        # print("CHChunker: self.merge_list_items...")
        if self.merge_list_items and list_items:  # need to yield
            yield DocChunk(
                text=self.delim.join([i.text for i in list_items]),
                meta=DocMeta(
                    doc_items=list_items,
                    headings=[heading_by_level[k] for k in sorted(heading_by_level)]
                    or None,
                    origin=dl_doc.origin,
                ),
            )

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

    def _have_text(self, item):
        return re.search("([\w]+)", item.text) is not None

    def _get_doc_chunk_heading(self, heading_by_level, prev_heading_by_level, item, captions, dl_doc):
        if prev_heading_by_level != {}:
            c = DocChunk(
                text="",
                meta=DocMeta(
                    doc_items=[item],
                    headings=[prev_heading_by_level[k] for k in sorted(prev_heading_by_level)]
                    or None,
                    captions=captions,
                    origin=dl_doc.origin,
                ),
            )
            return c
        else:
            return None

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

    def _get_prev_heading_by_level(self, item, heading_by_level, prev_heading_by_level, level):
        temp_heading_by_level = heading_by_level
        if level in heading_by_level and heading_by_level[level] != item.text and level == len(heading_by_level):
            # Take only headings up until this point.
            # If there are 3 level before this and new heading only have 1, take only 1.
            keys_to_del = [k for k in temp_heading_by_level if k > level]
            for k in keys_to_del:
                temp_heading_by_level.pop(k, None)

            prev_heading_by_level = temp_heading_by_level
        elif heading_by_level == {}:
            prev_heading_by_level[level] = item.text

        return prev_heading_by_level
