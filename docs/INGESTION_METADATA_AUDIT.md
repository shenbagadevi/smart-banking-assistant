# Ingestion Metadata Audit

## Scope

This audit focuses only on metadata quality in the ingestion pipeline and does not change implementation.

Relevant files:
- [src/ingestion/parser.py](../src/ingestion/parser.py)
- [src/ingestion/chunker.py](../src/ingestion/chunker.py)
- [src/ingestion/metadata_enrichment.py](../src/ingestion/metadata_enrichment.py)
- [src/api/v1/services/ingestion_service.py](../src/api/v1/services/ingestion_service.py)

---

## Executive summary

The ingestion pipeline is producing product metadata at the wrong scope and attaching it too broadly across elements. The root problem is that document-level product inference is applied to every parsed element before chunking, and then chunk metadata inherits that document-wide state repeatedly. This causes:

1. `image_caption` chunks to receive banking product metadata even when the image content is unrelated.
2. section and heading metadata to carry merged or stale product context across multiple products.
3. `loan_type` and `product_category` to be assigned based on a whole-document keyword scan instead of section-level or chunk-level evidence.
4. `source_page` to become null when page provenance is missing or when converted document structure loses page metadata.

The result is noisy retrieval, incorrect document attribution, and broken citation display in demo conditions.

---

## 1) Why image_caption chunks are getting banking product metadata

### Root cause

`parse_document()` in [src/ingestion/parser.py](../src/ingestion/parser.py) does this:

- calls `_infer_product_metadata(file_path.name, parsed_elements)`
- then loops through every element and executes:
  `element["metadata"].update(product_category)`

This is a global document-level assignment applied to all elements, including `image_caption` records created later in `extract_all_elements()`.

The metadata function itself is also global and doc-wide:

- `_infer_product_metadata()` in [src/ingestion/parser.py](../src/ingestion/parser.py) concatenates `document_name` + all element contents into a single text string
- it then scans the whole string for product phrases like `home loan`, `personal loan`, `credit card`
- it assigns one product set to the entire document, regardless of whether a chunk is a chart image, figure caption, or unrelated section

Because every element carries the same `loan_type` / `product_name` / `product_category`, image captions are not isolated. They inherit the file-level product label even when the actual image is unrelated or only a chart/figure from a different part of the document.

### Affected functions

- [src/ingestion/parser.py](../src/ingestion/parser.py): `parse_document()`
- [src/ingestion/parser.py](../src/ingestion/parser.py): `_infer_product_metadata()`
- [src/ingestion/parser.py](../src/ingestion/parser.py): `extract_all_elements()` / image block

### Exact fix locations

- In `parse_document()`, avoid applying document-wide product metadata directly to every parsed element.
- In `_infer_product_metadata()`, infer product metadata by section or by a smaller local context, not from the entire document text.
- In the image-processing branch inside `extract_all_elements()`, preserve image-only metadata and do not inherit a file-wide product label unless the image caption explicitly matches that section.

### Demo impact

Image chunks appear as if they were core product policy text. That pollutes retrieval with irrelevant image metadata and causes incorrect answers/citations during the live demo.

---

## 2) Why heading/section contains multiple unrelated products

### Root cause

The heading/section metadata is carried through a document-wide state model that is not reset correctly per section and product.

In [src/ingestion/parser.py](../src/ingestion/parser.py), `extract_all_elements()` uses mutable variables:

- `current_heading`
- `current_section`

These are updated as the parser walks the document, but this logic is not product-aware. When a document contains multiple banking products or multiple policy groups, the parser may keep a prior heading/section active across elements from a different product context.

Then in [src/ingestion/chunker.py](../src/ingestion/chunker.py):

- `prepare_chunks()` walks all normalized elements and updates `current_heading` / `current_section` as inherited state
- if the heading is valid and present, it is saved into `meta["heading"]` and `meta["section"]`
- then `build_chunk()` copies the inherited metadata into each chunk

This means a chunk can inherit a previously seen heading/section that belongs to a different product, especially when a document mixes multiple loan products or policy sections, or when DOCX/PDF conversion loses original heading semantics.

### Additional problem

The document-level metadata assignment happens before chunking, and a product label is reused for all elements. Once the product label is attached to the whole element set, the subsequent heading/section inheritance logic keeps that product context alive across unrelated blocks.

### Affected functions

- [src/ingestion/parser.py](../src/ingestion/parser.py): `extract_all_elements()`
- [src/ingestion/chunker.py](../src/ingestion/chunker.py): `prepare_chunks()`
- [src/ingestion/chunker.py](../src/ingestion/chunker.py): `build_chunk()`

### Exact fix locations

- In `extract_all_elements()`, reset heading/section state when the structural context changes, and avoid carrying a previous section across unrelated product blocks.
- In `prepare_chunks()`, restrict inheritance of heading/section to the same product section context instead of global previous values.
- In `build_chunk()`, ensure heading and section are only copied when they are actually valid for that specific chunk and not inherited from unrelated content.

### Demo impact

The user sees retrieval chunks labeled with mixed products like home loan, personal loan, and credit card in the same section context, which damages relevance and makes the assistant look unreliable.

---

## 3) Why loan_type/product_category are incorrectly assigned

### Root cause

The assignment logic is document-wide and simplistic.

In [src/ingestion/parser.py](../src/ingestion/parser.py):

```python
text = document_name.lower()
text += " " + " ".join(element["content"].lower() for element in parsed_elements)
```

Then:

```python
if "home loan" in text:
    metadata["product_category"] = "loan"
    metadata["loan_type"] = "home_loan"
    metadata["product_name"] = "Home Loan"
elif "personal loan" in text:
    ...
```

This logic makes a single decision for the whole document. It does not check whether the detected phrase belongs to the current element or section. If the document contains multiple product references, the first match wins and all elements inherit that label.

The helper in [src/ingestion/metadata_enrichment.py](../src/ingestion/metadata_enrichment.py) is also a simple first-match lookup and not section-aware. It is a lightweight classifier, not a safe place to assign document-level product metadata.

This makes product categorization especially brittle in mixed or multi-product marketing/knowledge documents.

### Affected functions

- [src/ingestion/parser.py](../src/ingestion/parser.py): `_infer_product_metadata()`
- [src/ingestion/metadata_enrichment.py](../src/ingestion/metadata_enrichment.py): `identify_product_category()`
- [src/ingestion/parser.py](../src/ingestion/parser.py): `parse_document()`

### Exact fix locations

- Replace document-level `_infer_product_metadata()` logic with section-aware or chunk-aware detection.
- Do not assign `loan_type` / `product_category` to every element after scanning the entire document.
- Use section classification only when there is clear evidence for that section; otherwise leave these fields empty rather than forcing a wrong value.
- Keep the helper in `metadata_enrichment.py` only as a lightweight fallback, not a global document assignment mechanism.

### Demo impact

The assistant may claim a home-loan policy when the actual chunk is about a credit card or a different loan product. This undermines trust and makes the demo look inconsistent.

---

## 4) Why source_page becomes null

### Root cause

`source_page` depends entirely on reliable page provenance from Docling items.

In [src/ingestion/parser.py](../src/ingestion/parser.py), `_get_page_number(item)` checks:

- `item.page_no`
- `item.prov[0].page_no`
- `item.prov[0].page`

If these are missing, it returns `None`.

That is exactly where nulls originate. Several extraction branches do not enforce a fallback:

- text elements set `"source_page": _get_page_number(item)`
- table elements set `"source_page": _get_page_number(item)`
- image captions set `"source_page": _get_page_number(item)`

Then in [src/ingestion/chunker.py](../src/ingestion/chunker.py), `build_chunk()` keeps this exact value:

```python
"source_page": element.get("metadata", {}).get("source_page")
```

If the parser never resolved the page, the field stays null. This commonly happens with docx conversion or image/caption elements whose item provenance is incomplete.

### Affected functions

- [src/ingestion/parser.py](../src/ingestion/parser.py): `_get_page_number()`
- [src/ingestion/parser.py](../src/ingestion/parser.py): `extract_all_elements()`
- [src/ingestion/parser.py](../src/ingestion/parser.py): `extract_image_elements()`
- [src/ingestion/chunker.py](../src/ingestion/chunker.py): `build_chunk()`

### Exact fix locations

- Improve `_get_page_number()` to recover page information from alternate provenance values if available.
- When provenance is missing, add a conservative fallback strategy rather than leaving nulls silently.
- In `build_chunk()`, validate that `source_page` is present before persisting a chunk as citation-worthy content.

### Demo impact

Citations and source display degrade badly when page numbers are null. UI-level source references become empty, and the answer appears ungrounded even when content was retrieved successfully.

---

## Why the metadata quality problem is amplified by the service layer

The ingestion service in [src/api/v1/services/ingestion_service.py](../src/api/v1/services/ingestion_service.py) logs metadata quality but does not correct it:

- it counts chunks with heading/section
- it logs metadata quality summary
- it does not validate that product metadata is section-accurate or that page provenance is non-null

This makes the bug visible in logs but not prevented before storage. In other words, the pipeline accepts bad metadata and stores it.

---

## Final assessment

### Root cause in one sentence

The pipeline applies document-wide product inference to all parsed elements, then carries that global metadata through section inheritance and chunk construction without scoping it to the actual element, product section, or page provenance.

### Demo impact

- Wrong product labels in the retrieval corpus
- Mixed section metadata across multiple products
- Broken or missing page citations
- Unreliable demo answers and poor trust signal

This is a metadata-quality issue in the ingestion pipeline, not a SQL or LLM-routing issue.
