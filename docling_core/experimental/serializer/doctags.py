"""Define classes for Doctags serialization."""

import html
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import AnyUrl, BaseModel
from typing_extensions import override

from docling_core.experimental.serializer.base import (
    BaseDocSerializer,
    BaseFallbackSerializer,
    BaseFormSerializer,
    BaseInlineSerializer,
    BaseKeyValueSerializer,
    BaseListSerializer,
    BasePictureSerializer,
    BaseTableSerializer,
    BaseTextSerializer,
    SerializationResult,
)
from docling_core.experimental.serializer.common import CommonParams, DocSerializer
from docling_core.types.doc.document import (
    CodeItem,
    DocItem,
    DoclingDocument,
    Formatting,
    FormItem,
    InlineGroup,
    KeyValueItem,
    ListItem,
    NodeItem,
    OrderedList,
    PictureClassificationData,
    PictureItem,
    PictureMoleculeData,
    TableItem,
    TextItem,
    UnorderedList,
)
from docling_core.types.doc.tokens import DocumentToken


def _wrap(text: str, wrap_tag: str) -> str:
    return f"<{wrap_tag}>{text}</{wrap_tag}>"


class DocTagsParams(CommonParams):
    """DocTags-specific serialization parameters."""

    class Mode(str, Enum):
        """DocTags serialization mode."""

        MINIFIED = "minified"
        HUMAN_FRIENDLY = "human_friendly"

    xsize: int = 500
    ysize: int = 500
    add_location: bool = True
    add_caption: bool = True
    add_content: bool = True
    add_table_cell_location: bool = False
    add_table_cell_text: bool = True
    add_page_break: bool = True

    mode: Mode = Mode.HUMAN_FRIENDLY


def _get_delim(params: DocTagsParams) -> str:
    if params.mode == DocTagsParams.Mode.HUMAN_FRIENDLY:
        delim = "\n"
    elif params.mode == DocTagsParams.Mode.MINIFIED:
        delim = ""
    else:
        raise RuntimeError(f"Unknown DocTags mode: {params.mode}")
    return delim


class DocTagsTextSerializer(BaseModel, BaseTextSerializer):
    """DocTags-specific text item serializer."""

    @override
    def serialize(
        self,
        *,
        item: TextItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        from docling_core.types.doc.document import SectionHeaderItem

        params = DocTagsParams(**kwargs)
        wrap_tag: Optional[str] = DocumentToken.create_token_name_from_doc_item_label(
            label=item.label,
            **({"level": item.level} if isinstance(item, SectionHeaderItem) else {}),
        )
        parts: list[str] = []

        if params.add_location:
            location = item.get_location_tokens(
                doc=doc,
                xsize=params.xsize,
                ysize=params.ysize,
            )
            if location:
                parts.append(location)

        if params.add_content:
            text_part = item.text
            text_part = doc_serializer.post_process(
                text=text_part,
                escape_html=False,  # TODO review
                formatting=item.formatting,
                hyperlink=item.hyperlink,
            )

            if isinstance(item, CodeItem):
                language_token = DocumentToken.get_code_language_token(
                    code_language=item.code_language,
                )
                text_part = f"{language_token}{text_part}"
            else:
                text_part = text_part.strip()
                if isinstance(item, ListItem):
                    wrap_tag = None  # deferring list item tags to list handling

            if text_part:
                parts.append(text_part)

        res = "".join(parts)
        if wrap_tag is not None:
            res = _wrap(text=res, wrap_tag=wrap_tag)
        return SerializationResult(text=res)


class DocTagsTableSerializer(BaseTableSerializer):
    """DocTags-specific table item serializer."""

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
        params = DocTagsParams(**kwargs)

        body = ""

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            if params.add_location:
                body += item.get_location_tokens(
                    doc=doc,
                    xsize=params.xsize,
                    ysize=params.ysize,
                )

            body += item.export_to_otsl(
                doc=doc,
                add_cell_location=params.add_table_cell_location,
                add_cell_text=params.add_table_cell_text,
                xsize=params.xsize,
                ysize=params.ysize,
            )

        if params.add_caption and len(item.captions):
            text = doc_serializer.serialize_captions(item, **kwargs).text

            if len(text):
                body += f"<{DocumentToken.CAPTION.value}>"
                for caption in item.captions:
                    if caption.cref not in doc_serializer.get_excluded_refs(**kwargs):
                        if isinstance(cap := caption.resolve(doc), DocItem):
                            body += cap.get_location_tokens(
                                doc=doc,
                                xsize=params.xsize,
                                ysize=params.ysize,
                            )
                body += f"{text.strip()}"
                body += f"</{DocumentToken.CAPTION.value}>"

        if body:
            body = _wrap(text=body, wrap_tag=DocumentToken.OTSL.value)

        return SerializationResult(text=body)


class DocTagsPictureSerializer(BasePictureSerializer):
    """DocTags-specific picture item serializer."""

    @override
    def serialize(
        self,
        *,
        item: PictureItem,
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = DocTagsParams(**kwargs)
        parts: list[str] = []

        if item.self_ref not in doc_serializer.get_excluded_refs(**kwargs):
            body = ""
            if params.add_location:
                body += item.get_location_tokens(
                    doc=doc,
                    xsize=params.xsize,
                    ysize=params.ysize,
                )

            classifications = [
                ann
                for ann in item.annotations
                if isinstance(ann, PictureClassificationData)
            ]
            if len(classifications) > 0:
                predicted_class = classifications[0].predicted_classes[0].class_name
                body += DocumentToken.get_picture_classification_token(predicted_class)

            smiles_annotations = [
                ann for ann in item.annotations if isinstance(ann, PictureMoleculeData)
            ]
            if len(smiles_annotations) > 0:
                body += _wrap(
                    text=smiles_annotations[0].smi, wrap_tag=DocumentToken.SMILES.value
                )
            parts.append(body)

        if params.add_caption and len(item.captions):
            text = doc_serializer.serialize_captions(item, **kwargs).text

            if len(text):
                body = ""
                for caption in item.captions:
                    if caption.cref not in doc_serializer.get_excluded_refs(**kwargs):
                        if isinstance(cap := caption.resolve(doc), DocItem):
                            body += cap.get_location_tokens(
                                doc=doc,
                                xsize=params.xsize,
                                ysize=params.ysize,
                            )
                            body += f"{text.strip()}"
                if body:
                    body = _wrap(text=body, wrap_tag=DocumentToken.CAPTION.value)
                    parts.append(body)

        text = "".join(parts)
        if text:
            token = DocumentToken.create_token_name_from_doc_item_label(
                label=item.label
            )
            text = _wrap(text=text, wrap_tag=token)
        return SerializationResult(text=text)


class DocTagsKeyValueSerializer(BaseKeyValueSerializer):
    """DocTags-specific key-value item serializer."""

    @override
    def serialize(
        self,
        *,
        item: KeyValueItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        params = DocTagsParams(**kwargs)

        body = ""

        page_no = 1
        if len(item.prov) > 0:
            page_no = item.prov[0].page_no

        if params.add_location:
            body += item.get_location_tokens(
                doc=doc,
                xsize=params.xsize,
                ysize=params.ysize,
            )

        # mapping from source_cell_id to a list of target_cell_ids
        source_to_targets: Dict[int, List[int]] = {}
        for link in item.graph.links:
            source_to_targets.setdefault(link.source_cell_id, []).append(
                link.target_cell_id
            )

        for cell in item.graph.cells:
            cell_txt = ""
            if cell.prov is not None:
                if len(doc.pages.keys()):
                    page_w, page_h = doc.pages[page_no].size.as_tuple()
                    cell_txt += DocumentToken.get_location(
                        bbox=cell.prov.bbox.to_top_left_origin(page_h).as_tuple(),
                        page_w=page_w,
                        page_h=page_h,
                        xsize=params.xsize,
                        ysize=params.ysize,
                    )
            if params.add_content:
                cell_txt += cell.text.strip()

            if cell.cell_id in source_to_targets:
                targets = source_to_targets[cell.cell_id]
                for target in targets:
                    # TODO centralize token creation
                    cell_txt += f"<link_{target}>"

            # TODO centralize token creation
            tok = f"{cell.label.value}_{cell.cell_id}"
            cell_txt = _wrap(text=cell_txt, wrap_tag=tok)
            body += cell_txt

        body = _wrap(body, DocumentToken.KEY_VALUE_REGION.value)
        return SerializationResult(text=body)


class DocTagsFormSerializer(BaseFormSerializer):
    """DocTags-specific form item serializer."""

    @override
    def serialize(
        self,
        *,
        item: FormItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        # TODO add actual implementation
        text_res = ""
        return SerializationResult(text=text_res)


class DocTagsListSerializer(BaseModel, BaseListSerializer):
    """DocTags-specific list serializer."""

    indent: int = 4

    @override
    def serialize(
        self,
        *,
        item: Union[UnorderedList, OrderedList],
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        is_inline_scope: bool = False,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        my_visited = visited or set()
        params = DocTagsParams(**kwargs)
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level + 1,
            is_inline_scope=is_inline_scope,
            visited=my_visited,
            **kwargs,
        )
        delim = _get_delim(params=params)
        if parts:
            text_res = delim.join(
                [
                    _wrap(text=p.text, wrap_tag=DocumentToken.LIST_ITEM.value)
                    for p in parts
                ]
            )
            text_res = f"{text_res}{delim}"
            wrap_tag = (
                DocumentToken.ORDERED_LIST.value
                if isinstance(item, OrderedList)
                else DocumentToken.UNORDERED_LIST.value
            )
            text_res = _wrap(text=text_res, wrap_tag=wrap_tag)
        else:
            text_res = ""
        return SerializationResult(text=text_res)


class DocTagsInlineSerializer(BaseInlineSerializer):
    """DocTags-specific inline group serializer."""

    @override
    def serialize(
        self,
        *,
        item: InlineGroup,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        list_level: int = 0,
        visited: Optional[set[str]] = None,  # refs of visited items
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        my_visited = visited or set()
        params = DocTagsParams(**kwargs)
        parts = doc_serializer.get_parts(
            item=item,
            list_level=list_level,
            is_inline_scope=True,
            visited=my_visited,
            **kwargs,
        )
        wrap_tag = DocumentToken.INLINE.value
        delim = _get_delim(params=params)
        text_res = delim.join([p.text for p in parts if p.text])
        if text_res:
            text_res = f"{text_res}{delim}"
            text_res = _wrap(text=text_res, wrap_tag=wrap_tag)
        return SerializationResult(text=text_res)


class DocTagsFallbackSerializer(BaseFallbackSerializer):
    """DocTags-specific fallback serializer."""

    @override
    def serialize(
        self,
        *,
        item: NodeItem,
        doc_serializer: "BaseDocSerializer",
        doc: DoclingDocument,
        **kwargs,
    ) -> SerializationResult:
        """Serializes the passed item."""
        text_res = ""
        return SerializationResult(text=text_res)


class DocTagsDocSerializer(DocSerializer):
    """DocTags-specific document serializer."""

    text_serializer: BaseTextSerializer = DocTagsTextSerializer()
    table_serializer: BaseTableSerializer = DocTagsTableSerializer()
    picture_serializer: BasePictureSerializer = DocTagsPictureSerializer()
    key_value_serializer: BaseKeyValueSerializer = DocTagsKeyValueSerializer()
    form_serializer: BaseFormSerializer = DocTagsFormSerializer()
    fallback_serializer: BaseFallbackSerializer = DocTagsFallbackSerializer()

    list_serializer: BaseListSerializer = DocTagsListSerializer()
    inline_serializer: BaseInlineSerializer = DocTagsInlineSerializer()

    params: DocTagsParams = DocTagsParams()

    def post_process(
        self,
        text: str,
        *,
        escape_html: bool = True,
        formatting: Optional[Formatting] = None,
        hyperlink: Optional[Union[AnyUrl, Path]] = None,
        **kwargs,
    ) -> str:
        """Apply some text post-processing steps."""
        res = text
        if escape_html:
            res = html.escape(res, quote=False)
        res = super().post_process(
            text=res,
            formatting=formatting,
            hyperlink=hyperlink,
        )
        return res

    @override
    def serialize_page(self, parts: list[SerializationResult]) -> SerializationResult:
        """Serialize a page out of its parts."""
        delim = _get_delim(params=self.params)
        text_res = delim.join([p.text for p in parts])
        return SerializationResult(text=text_res)

    @override
    def serialize_doc(self, pages: list[SerializationResult]) -> SerializationResult:
        """Serialize a document out of its pages."""
        delim = _get_delim(params=self.params)
        if self.params.add_page_break:
            page_sep = f"{delim}<{DocumentToken.PAGE_BREAK.value}>{delim}"
            content = page_sep.join([p.text for p in pages if p.text])
        else:
            content = self.serialize_page(parts=pages).text
        wrap_tag = DocumentToken.DOCUMENT.value
        text_res = f"<{wrap_tag}>{content}{delim}</{wrap_tag}>"
        return SerializationResult(text=text_res)
