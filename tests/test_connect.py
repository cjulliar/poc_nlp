import unittest
from unittest.mock import patch, MagicMock
from app import utils

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

if __name__ == '__main__':
    unittest.main()
