# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Trần Hải Ninh
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Nghĩa là vector embedding của hai đoạn văn bản tạo góc rất nhỏ (gần 0°) với nhau trong không gian nhiều chiều — hai văn bản đang truyền đạt cùng một khái niệm hoặc ý nghĩa cốt lõi dù có thể dùng từ ngữ hoàn toàn khác nhau. Giá trị cosine similarity bằng 1.0 nghĩa là hai vector hoàn toàn trùng hướng; bằng 0 nghĩa là không liên quan; âm nghĩa là ngữ nghĩa đối lập.

**Ví dụ HIGH similarity:**
- Sentence A: "An emergency fund should cover three to six months of living expenses."
- Sentence B: "You should save enough money to last several months if you lose your job."
- Tại sao tương đồng: Cả hai câu đều diễn đạt cùng một khuyến nghị tài chính (quỹ dự phòng = vài tháng chi phí sinh hoạt), chỉ khác cách diễn đạt bề mặt — model embedding nhận ra sự tương đồng ngữ nghĩa sâu bên dưới.

**Ví dụ LOW similarity:**
- Sentence A: "Compound interest earns returns on your previous returns over time."
- Sentence B: "The best recipe for banana bread uses ripe bananas and brown sugar."
- Tại sao khác: Hai câu hoàn toàn thuộc hai domain không liên quan (tài chính vs nấu ăn). Vector embedding của chúng gần như vuông góc trong không gian embedding.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đánh giá **hướng** của vector, không phải **độ dài**. Với text, một đoạn văn dài và đoạn văn ngắn cùng nói về một chủ đề sẽ có vector có magnitude khác nhau nhiều nhưng hướng gần như giống nhau — cosine similarity sẽ cho kết quả cao còn Euclidean distance sẽ cho kết quả không chính xác vì bị ảnh hưởng bởi số lượng từ. Ngoài ra, khi embeddings được normalize về unit sphere (như `text-embedding-3-small`), cosine similarity tương đương với dot product và tính toán nhanh hơn.

---

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

> **Phép tính:**
> ```
> step       = chunk_size - overlap = 500 - 50 = 450
> num_chunks = ceil((doc_length - overlap) / (chunk_size - overlap))
>            = ceil((10,000 - 50) / 450)
>            = ceil(9,950 / 450)
>            = ceil(22.11)
>            = 23 chunks
> ```
> **Đáp án: 23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap tăng lên 100: `ceil((10,000 - 100) / 400) = ceil(9,900 / 400) = ceil(24.75) = 25 chunks` — tăng thêm 2 chunks. Overlap lớn hơn đảm bảo rằng mỗi câu, kể cả câu nằm ngay tại ranh giới của hai chunk, sẽ xuất hiện đầy đủ trong ít nhất một chunk. Điều này đặc biệt quan trọng với tài liệu tài chính — một câu bị cắt đứt giữa định nghĩa và ví dụ có thể làm mất đi ý nghĩa quan trọng khi retrieval.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Personal Finance & Investing for Students (Tài chính cá nhân và đầu tư dành cho sinh viên)

**Tại sao nhóm chọn domain này?**
> Tài chính cá nhân là domain cực kỳ thực tế và hữu ích cho sinh viên nhưng lại ít được dạy chính thức trong trường học. Nội dung có cấu trúc rõ ràng theo chủ đề (budgeting, saving, investing, debt) và phù hợp để thử nghiệm metadata filtering theo `topic`. Quan trọng hơn, các câu hỏi trong domain này rất cụ thể và có câu trả lời xác định (ví dụ "emergency fund cần bao nhiêu tháng?") — lý tưởng để đánh giá retrieval precision một cách khách quan.

---

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | personal_budgeting.md | data/personal_budgeting.md | 2,666 | `topic: budgeting`, `difficulty: beginner`, `lang: en` |
| 2 | savings_strategies.md | data/savings_strategies.md | 2,680 | `topic: saving`, `difficulty: beginner`, `lang: en` |
| 3 | investing_basics.md | data/investing_basics.md | 3,263 | `topic: investing`, `difficulty: intermediate`, `lang: en` |
| 4 | debt_management.md | data/debt_management.md | 3,204 | `topic: debt`, `difficulty: intermediate`, `lang: en` |
| 5 | financial_goals_planning.md | data/financial_goals_planning.md | 3,592 | `topic: planning`, `difficulty: beginner`, `lang: en` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `topic` | string | `"budgeting"`, `"saving"`, `"investing"`, `"debt"`, `"planning"` | Filter theo chủ đề tài chính cụ thể — khi user hỏi về nợ thì không cần tìm trong tài liệu về đầu tư, giảm false positives đáng kể |
| `difficulty` | string | `"beginner"`, `"intermediate"` | Giúp hệ thống điều chỉnh kết quả theo trình độ người dùng — sinh viên mới bắt đầu nhận tài liệu `beginner`, người đã có kiến thức cơ bản nhận `intermediate` |
| `lang` | string | `"en"`, `"vi"` | Hỗ trợ filter ngôn ngữ khi mở rộng thêm tài liệu tiếng Việt — query tiếng Việt chỉ tìm trong `lang: vi` |
| `source` | string | `"data/investing_basics.md"` | Truy xuất nguồn gốc file để cite và verify tính chính xác của câu trả lời |
| `doc_id` | string | `"investing_basics"` | ID duy nhất để `delete_document` và group chunks của cùng một tài liệu |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` với `chunk_size=300` trên 2 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| personal_budgeting.md | `fixed_size` | 10 | 294 | Trung bình — hay cắt giữa định nghĩa và ví dụ |
| personal_budgeting.md | `by_sentences` | 8 | 331 | Tốt — giữ nguyên câu, ý tài chính đầy đủ |
| personal_budgeting.md | `recursive` | 15 | 176 | Tốt — bám theo cấu trúc heading/paragraph |
| investing_basics.md | `fixed_size` | 12 | 299 | Trung bình |
| investing_basics.md | `by_sentences` | 11 | 295 | Tốt — mỗi câu tài chính là một fact hoàn chỉnh |
| investing_basics.md | `recursive` | 17 | 190 | Tốt — phân tách rõ theo section |

### Strategy Của Tôi

**Loại:** `SentenceChunker` với `max_sentences_per_chunk=2`

**Mô tả cách hoạt động:**
> `SentenceChunker` dùng regex `(?<=[.!?])\s+` với lookbehind assertion để tách text tại khoảng trắng sau dấu ngắt câu, giữ dấu chấm gắn với câu (không bị tách rời). Sau đó nhóm `max_sentences_per_chunk=2` câu liên tiếp vào một chunk. Tôi chọn 2 câu/chunk — đủ ngữ cảnh để câu thứ nhất giới thiệu khái niệm và câu thứ hai giải thích — mà vẫn đủ nhỏ để retrieval chính xác.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tài liệu tài chính cá nhân viết theo style "fact → explanation": mỗi câu là một khuyến nghị hoặc định nghĩa độc lập và tự hoàn chỉnh. Câu "The debt avalanche method prioritizes paying off the debt with the highest interest rate first" mang đủ thông tin để trả lời query liên quan mà không cần thêm context từ câu trước. `SentenceChunker` khai thác đặc điểm này tốt hơn `RecursiveChunker` — vốn ưu tiên paragraph breaks và tạo ra chunks bao gồm nhiều fact không liên quan trong cùng một paragraph dài.

**Code snippet (custom max_sentences):**
```python
from src.chunking import SentenceChunker

# 2 sentences per chunk — balance between precision and context
chunker = SentenceChunker(max_sentences_per_chunk=2)
chunks = chunker.chunk(document_content)
# Result: each chunk = 1 financial fact + 1 supporting explanation
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| investing_basics.md | Best baseline: `recursive` | 17 | 190 chars | Tốt — tách sạch theo section heading |
| investing_basics.md | **`by_sentences` max=2 (của tôi)** | **22** | **148 chars** | **Tốt hơn — mỗi chunk là 1 fact tài chính cụ thể, score truy xuất cao hơn cho query chi tiết** |

> `SentenceChunker` tạo nhiều chunk nhỏ hơn `recursive`, nhưng mỗi chunk tập trung hơn vào một khái niệm duy nhất. Với domain tài chính, nơi user hỏi câu cụ thể ("What is compound interest?"), granularity cao này mang lại precision tốt hơn.

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Chunk Count | Avg Length | Retrieval Quality | Điểm mạnh | Điểm yếu |
|-----------|----------|-------------|------------|-------------------|-----------|----------|
| **Tôi (Ning)** | SentenceChunker (max_sentences=2) | 22 | ~148 chars | Tốt — precision cao với query cụ thể | Chunk là unit ngữ nghĩa hoàn chỉnh, dễ hiểu | Chunk quá nhỏ khi query cần tổng hợp nhiều ý |
| **Minh Anh** | RecursiveChunker (chunk_size=400) | 14 | ~230 chars | Tốt — bám cấu trúc tài liệu | Giữ nguyên cấu trúc heading/paragraph | Chunk có thể span nhiều sub-topic trong cùng section |
| **Hải Đăng** | FixedSizeChunker (chunk_size=350, overlap=80) | 12 | ~310 chars | Trung bình — overlap giúp không bỏ sót fact | Đơn giản, dễ tune | Cắt ngang câu định nghĩa tài chính, mất nghĩa |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với domain tài chính cá nhân, `SentenceChunker` và `RecursiveChunker` đều cho kết quả tốt hơn `FixedSizeChunker`. Tuy nhiên xét riêng cho loại query cụ thể ("What is X?", "How does Y work?"), `SentenceChunker` với max=2 thắng vì mỗi fact tài chính được đóng gói vào một chunk độc lập — retrieval đưa về đúng định nghĩa cần tìm thay vì một paragraph dài chứa nhiều khái niệm không liên quan.

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Dùng `re.split(r'(?<=[.!?])\s+')` với lookbehind assertion — split tại khoảng trắng ngay sau dấu ngắt câu, giữ dấu chấm gắn với câu gốc. Strip whitespace từng câu, lọc bỏ chuỗi rỗng. Sau đó loop qua list sentences theo step `max_sentences_per_chunk`, join nhóm lại bằng dấu cách. Edge case: text rỗng → `[]`; text không có dấu ngắt câu → `[text]` (toàn bộ text là một chunk duy nhất).

**`RecursiveChunker.chunk` / `_split`** — approach:
> `chunk()` là entry point gọi `_split(text, self.separators)` với toàn bộ separator list. Trong `_split()`, base case: `len(text) <= chunk_size` → return `[text]`; không còn separator (hoặc separator là `""`) → force-split theo character. Trường hợp chính: tìm separator hiện tại trong text, tách thành parts, **greedy-merge** lại: gộp liên tiếp các parts miễn là tổng length + sep vẫn `<= chunk_size`. Chunk kết quả nào vẫn quá lớn → đệ quy với `remaining_separators[1:]`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `_make_record()` nhúng `doc.content` qua `self._embedding_fn`, tạo dict `{content, metadata (+ doc_id), embedding}`. `add_documents()` duyệt từng doc, gọi `_make_record()` và append vào `self._store`. `search()` ủy quyền cho `_search_records()`: embed query → tính dot product với từng stored embedding → sort descending → slice top_k. Vì `text-embedding-3-small` trả về unit-normalized vectors, dot product = cosine similarity, không cần chia thêm magnitude.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter()` **filter trước, search sau**: list comprehension lọc `self._store` chỉ giữ records mà mọi key-value trong `metadata_filter` đều khớp, sau đó chạy `_search_records()` trên tập đã lọc. Chiến lược này giảm số lượng dot product cần tính. `delete_document()` dùng list comprehension loại bỏ records có `metadata['doc_id'] == doc_id`, so sánh `len(before)` và `len(after)` để return `True/False`.

### KnowledgeBaseAgent

**`answer`** — approach:
> Retrieve top_k chunks từ store. Build prompt theo 3-section template: (1) System instruction "Answer based only on context", (2) Context block với từng chunk đánh số `[1] ... [2] ...` để LLM có thể tham chiếu rõ ràng, (3) Question và `Answer:` suffix kích hoạt generation mode. Cách đánh số chunk giúp LLM trace-back được nguồn khi cần explain, và instruction "based only on context" giảm hallucination.

### Test Results

```
platform win32 -- Python 3.13.7, pytest-9.0.3
rootdir: E:\VinUni\assignments\Day-07-Lab-Data-Foundations
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

42 passed in 0.15s
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

> **Embedder sử dụng:** `OpenAIEmbedder` (`text-embedding-3-small`) — semantic embeddings thực tế, không phải mock.

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | An emergency fund should cover three to six months of living expenses. | You should save enough money to last several months if you lose your job. | high | 0.5945 | Đúng |
| 2 | Compound interest earns returns on your previous returns over time. | The best recipe for banana bread uses ripe bananas and brown sugar. | low | 0.0783 | Đúng |
| 3 | Stocks represent ownership in a company and carry high risk. | Equities are ownership stakes in businesses that can fluctuate significantly in value. | high | 0.6076 | Đúng |
| 4 | The debt avalanche method pays off the highest interest rate debt first. | Paying down the most expensive debt reduces total interest paid over time. | high | 0.7035 | Đúng |
| 5 | Index funds provide diversification by tracking an entire market index. | Student loan interest capitalizes if left unpaid during the deferment period. | low | 0.1334 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 4 bất ngờ nhất với score cao nhất (0.7035): hai câu không dùng từ nào trùng nhau ("debt avalanche method" vs "paying down the most expensive debt"), nhưng embedding model nhận ra chúng đang nói về cùng một hành động tài chính. Điều này cho thấy `text-embedding-3-small` không hoạt động theo kiểu matching keyword — nó encode **khái niệm** và **intent** vào vector space, khiến paraphrase ngữ nghĩa (dù từ ngữ khác hoàn toàn) có embedding gần nhau. Pair 1 cũng ấn tượng: "emergency fund" và "save money in case of job loss" không có keyword chung nhưng vẫn đạt 0.59 — embedding hiểu được mục đích phía sau ngôn ngữ bề mặt.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries trên implementation cá nhân với `OpenAIEmbedder` và `EmbeddingStore` (in-memory). Documents lưu nguyên file (không chunk trước), 5 docs trong store.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | How much should I save for an emergency fund? | 3–6 tháng chi phí sinh hoạt trong tài khoản tiết kiệm thanh khoản cao |
| 2 | What is compound interest and how does it grow my money? | Lãi kép là lãi tính trên cả gốc lẫn lãi đã tích lũy — $1,000 ở 7%/năm tăng lên $7,600 sau 30 năm |
| 3 | How do I create a monthly budget step by step? | 5 bước: tính thu nhập → liệt kê chi phí cố định → ước tính chi phí biến đổi → trừ → điều chỉnh và theo dõi hàng tuần |
| 4 | What is the debt avalanche method? | Ưu tiên trả nợ có lãi suất cao nhất trước, tối thiểu hóa tổng lãi phải trả |
| 5 | Should I invest in index funds or individual stocks? | Index funds được khuyến nghị cho người mới — diversification rộng, phí thấp, không cần phân tích cổ phiếu riêng lẻ |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | How much should I save for an emergency fund? | savings_strategies.md — "3–6 months of expenses in a HYSA" | 0.5722 | Yes | Cần 3–6 tháng chi phí trong HYSA; freelancer hoặc ngành bấp bênh nên chọn 6 tháng |
| 2 | What is compound interest and how does it grow my money? | investing_basics.md — "$1,000 at 7% grows to $7,600 over 30 years" | 0.4409 | Yes | Compound interest tính lãi trên lãi đã tích lũy — bắt đầu sớm quan trọng hơn đầu tư nhiều nhưng muộn |
| 3 | How do I create a monthly budget step by step? | personal_budgeting.md — "5-step budgeting process" | 0.6362 | Yes | 5 bước: tính thu nhập → fixed expenses → variable expenses → surplus/deficit → track weekly |
| 4 | What is the debt avalanche method? (filter: topic=debt) | debt_management.md — "Avalanche: highest interest rate first" | 0.6255 | Yes | Trả nợ lãi suất cao nhất trước trong khi thanh toán tối thiểu các khoản khác — tối ưu về tổng lãi phải trả |
| 5 | Should I invest in index funds or individual stocks? (filter: topic=investing) | investing_basics.md — "Index funds: diversification at low cost" | 0.4865 | Yes | Index funds được khuyến nghị cho người mới: diversification tức thì, expense ratio thấp, không cần chọn cổ phiếu |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

> **Nhận xét:** Dùng `OpenAIEmbedder` kết hợp metadata filter (Q4, Q5) cho precision hoàn hảo 5/5. Query 4 dùng `search_with_filter(metadata_filter={"topic": "debt"})` — không có filter, document `investing_basics` có thể tranh score với `debt_management` vì cả hai đề cập đến interest rates. Metadata filtering đã loại bỏ hoàn toàn noise này.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Minh Anh cho thấy rằng `RecursiveChunker` với heading-aware separator (`## `) thêm vào đầu danh sách hiệu quả hơn default separators cho tài liệu markdown — heading tạo ra natural boundary giữa các sub-topic tài chính khác nhau. Điều này khiến tôi nhận ra separators mặc định không phải tối ưu cho mọi domain, và việc customize separators là một trong những cải tiến dễ nhất mà lại có impact cao nhất.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một nhóm dùng domain luật pháp cho thấy tầm quan trọng của metadata cực kỳ granular (`dieu`, `khoan`, `chuong`) — khi tài liệu có nhiều đơn vị nhỏ tương tự nhau (như các khoản luật), filter theo số điều cụ thể loại bỏ gần như toàn bộ false positives. Đây là insight quan trọng: đôi khi metadata design quan trọng hơn chunking strategy, vì metadata tốt có thể bù đắp cho chunking chưa hoàn hảo.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Thứ nhất, sẽ pre-chunk từng tài liệu theo section (dùng `##` heading làm boundary) trước khi tạo `Document` — thêm metadata `section` như `"emergency_fund"`, `"compound_interest"` để filter cực kỳ chính xác. Thứ hai, sẽ thêm metadata `difficulty_score` là số thực (0.0–1.0) thay vì categorical string để hỗ trợ range query. Thứ ba, sẽ thử **HyDE (Hypothetical Document Embeddings)** — cho LLM viết đoạn trả lời giả định từ query rồi embed đoạn đó thay vì embed câu hỏi — vì tài liệu tài chính thường trả lời theo format khác với query.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| **Tổng** | | **87 / 100** |
