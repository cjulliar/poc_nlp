import unittest
from unittest.mock import patch, MagicMock
from apps.single_support import utils

class TestUtils(unittest.TestCase):

    # Test le chargement de WikipediaLoader avec un vrai page_name
    # Test sur la division du document par RecursiveCharacterTextSplitter
    # Test le retriever retourne par Chroma
    @patch('app.utils.WikipediaLoader')
    @patch('app.utils.RecursiveCharacterTextSplitter')
    @patch('app.utils.Chroma')
    def test_preprocess_wikipedia_functional(self, mock_chroma, mock_text_splitter, mock_wikipedia_loader):
        # Mock WikipediaLoader
        mock_loader_instance = mock_wikipedia_loader.return_value
        mock_loader_instance.load.return_value = ["Mocked Documents"]

        # Mock RecursiveCharacterTextSplitter
        mock_splitter_instance = mock_text_splitter.return_value
        mock_splitter_instance.split_documents.return_value = ["Split Document 1", "Split Document 2"]

        # Mock Chroma
        mock_chroma_instance = MagicMock()
        mock_chroma.from_documents.return_value = mock_chroma_instance
        mock_retriever = MagicMock()
        mock_chroma_instance.as_retriever.return_value = mock_retriever

        retriever = utils.preprocess_wikipedia("test_page", "Mocked Embedding")

        self.assertEqual(retriever, mock_retriever)
        mock_wikipedia_loader.assert_called_once_with(query="test_page", load_max_docs=1, doc_content_chars_max=10000)
        mock_text_splitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        mock_chroma.from_documents.assert_called_once_with(["Split Document 1", "Split Document 2"], "Mocked Embedding")

    # Test le chargement de YoutubeLoader avec une vraie URL
    # Test sur la division du document par RecursiveCharacterTextSplitter
    # Test le retriever retourne par Chroma
    @patch('app.utils.YoutubeLoader')
    @patch('app.utils.RecursiveCharacterTextSplitter')
    @patch('app.utils.Chroma')
    def test_preprocess_youtube_functional(self, mock_chroma, mock_text_splitter, mock_youtube_loader):
        # Mock YoutubeLoader
        mock_loader_instance = mock_youtube_loader.return_value
        mock_loader_instance.load.return_value = [MagicMock(page_content="Mocked Transcript")]

        # Mock RecursiveCharacterTextSplitter
        mock_splitter_instance = mock_text_splitter.return_value
        mock_splitter_instance.split_documents.return_value = ["Split Document 1", "Split Document 2"]

        # Mock Chroma
        mock_chroma_instance = MagicMock()
        mock_chroma.from_documents.return_value = mock_chroma_instance
        mock_retriever = MagicMock()
        mock_chroma_instance.as_retriever.return_value = mock_retriever

        retriever = utils.preprocess_youtube("https://youtube.com/watch?v=test_id", "Mocked Embedding")

        self.assertEqual(retriever, mock_retriever)
        mock_youtube_loader.assert_called_once_with(video_id="test_id")
        mock_text_splitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200)
        mock_chroma.from_documents.assert_called_once_with(["Split Document 1", "Split Document 2"], "Mocked Embedding")

    # Test le chargement et la division de texte d'un PDF
    # Test sur la division du document par RecursiveCharacterTextSplitter
    # Test le retriever retourne par Chroma
    @patch('app.utils.extract_text_from_pdf')
    @patch('app.utils.RecursiveCharacterTextSplitter')
    @patch('app.utils.Chroma')
    def test_preprocess_pdf_functional(self, mock_chroma, mock_text_splitter, mock_extract_text):
        # Mock extract_text_from_pdf
        mock_extract_text.return_value = "Mocked PDF Text"

        # Mock RecursiveCharacterTextSplitter
        mock_splitter_instance = mock_text_splitter.return_value
        mock_splitter_instance.split_text.return_value = ["Split Text 1", "Split Text 2"]
        mock_splitter_instance.split_documents.return_value = ["Split Document 1", "Split Document 2"]

        # Mock Chroma
        mock_chroma_instance = MagicMock()
        mock_chroma.from_documents.return_value = mock_chroma_instance
        mock_retriever = MagicMock()
        mock_chroma_instance.as_retriever.return_value = mock_retriever

        retriever = utils.preprocess_pdf("test.pdf", "Mocked Embedding")

        self.assertEqual(retriever, mock_retriever)
        mock_extract_text.assert_called_once_with("test.pdf")
        mock_text_splitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        mock_chroma.from_documents.assert_called_once_with(["Split Document 1", "Split Document 2"], "Mocked Embedding")

    # Test le chargement et la division de texte d'un corpus
    # Test sur la division du document par RecursiveCharacterTextSplitter
    # Test le retriever retourne par PineconeVectorStore
    @patch('app.utils.load_documents_from_directory')
    @patch('app.utils.RecursiveCharacterTextSplitter')
    @patch('app.utils.PineconeVectorStore')
    def test_preprocess_corpus_functional(self, mock_pinecone, mock_text_splitter, mock_load_documents):
        # Mock load_documents_from_directory
        mock_load_documents.return_value = ["Mocked Corpus Text 1", "Mocked Corpus Text 2"]

        # Mock RecursiveCharacterTextSplitter
        mock_splitter_instance = mock_text_splitter.return_value
        mock_splitter_instance.split_text.side_effect = [
            ["Split Text 1.1", "Split Text 1.2"],
            ["Split Text 2.1", "Split Text 2.2"]
        ]
        mock_splitter_instance.split_documents.side_effect = [
            ["Split Document 1.1", "Split Document 1.2"],
            ["Split Document 2.1", "Split Document 2.2"]
        ]

        # Mock PineconeVectorStore
        mock_pinecone_instance = MagicMock()
        mock_pinecone.from_documents.return_value = mock_pinecone_instance
        mock_retriever = MagicMock()
        mock_pinecone_instance.as_retriever.return_value = mock_retriever

        retriever = utils.preprocess_corpus("test_directory", "Mocked Embedding")

        self.assertEqual(retriever, mock_retriever)
        mock_load_documents.assert_called_once_with("test_directory")
        mock_text_splitter.assert_called_with(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        mock_pinecone.from_documents.assert_called_once()
        mock_pinecone_instance.as_retriever.assert_called_once()

    # Tests pour extract_text_from_pdf et extract_text_from_docx
    @patch('app.utils.pymupdf.open')
    def test_extract_text_from_pdf_success(self, mock_open):
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Sample text"
        mock_pdf.load_page.return_value = mock_page
        mock_pdf.__len__.return_value = 1
        mock_open.return_value = mock_pdf

        text = utils.extract_text_from_pdf("dummy.pdf")
        self.assertEqual(text, "Sample text")

    @patch('app.utils.pymupdf.open')
    def test_extract_text_from_pdf_error(self, mock_open):
        mock_open.side_effect = Exception("Error reading PDF")
        text = utils.extract_text_from_pdf("dummy.pdf")
        self.assertEqual(text, "")

    @patch('app.utils.Document')
    def test_extract_text_from_docx_success(self, mock_document):
        mock_doc_instance = mock_document.return_value
        mock_doc_instance.paragraphs = [MagicMock(text="Paragraph 1"), MagicMock(text="Paragraph 2")]

        text = utils.extract_text_from_docx("dummy.docx")
        self.assertEqual(text, "Paragraph 1\nParagraph 2")

    @patch('app.utils.Document')
    def test_extract_text_from_docx_error(self, mock_document):
        mock_document.side_effect = Exception("Error reading DOCX")
        text = utils.extract_text_from_docx("dummy.docx")
        self.assertEqual(text, "")

    # Test pour load_documents_from_directory
    @patch('app.utils.os.listdir')
    @patch('app.utils.extract_text_from_pdf')
    @patch('app.utils.extract_text_from_docx')
    def test_load_documents_from_directory(self, mock_extract_docx, mock_extract_pdf, mock_listdir):
        mock_listdir.return_value = ["file1.pdf", "file2.docx", "file3.txt"]
        mock_extract_pdf.return_value = "PDF text"
        mock_extract_docx.return_value = "DOCX text"

        documents = utils.load_documents_from_directory("dummy_directory")
        self.assertEqual(documents, ["PDF text", "DOCX text"])


@patch('app.utils.PREPROC')
@patch('app.utils.embedding')
def test_define_source_single_file(self, mock_embedding, mock_preproc):
    mock_func = MagicMock()
    mock_preproc.__getitem__.return_value = mock_func
    file = MagicMock(name="file.pdf")
    
    utils.define_source(file, "pdf")

    mock_func.assert_called_once_with(file, mock_embedding)
    self.assertIn("file.pdf", utils.retriever_store)

    # Test pour define_source
    @patch('app.utils.PREPROC')
    @patch('app.utils.embedding')
    def test_define_source_multiple_files(self, mock_embedding, mock_preproc):
        mock_func = MagicMock()
        mock_preproc.__getitem__.return_value = mock_func
        files = [MagicMock(name="file1.pdf"), MagicMock(name="file2.docx")]
        
        utils.define_source(files, "corpus")

        mock_func.assert_any_call(files[0], mock_embedding)
        mock_func.assert_any_call(files[1], mock_embedding)
        self.assertIn("file1.pdf", utils.retriever_store)
        self.assertIn("file2.docx", utils.retriever_store)


    # Test pour query_llm
    @patch('app.utils.get_answer_llm')
    def test_query_llm_with_selected_source(self, mock_get_answer):
        mock_get_answer.return_value = "Mocked Answer"
        utils.retriever_store = {"test.pdf": MagicMock()}
        utils.username = "test_user"
        utils.source = "test.pdf"

        response = utils.query_llm("test query", [])
        self.assertEqual(response, "Mocked Answer")

    @patch('app.utils.get_answer_llm')
    def test_query_llm_without_selected_source(self, mock_get_answer):
        utils.retriever_store = {}
        utils.username = "test_user"
        utils.source = "nonexistent.pdf"

        response = utils.query_llm("test query", [])
        self.assertEqual(response, "Erreur: Aucun document sélectionné.")


if __name__ == '__main__':
    unittest.main()
