import unittest
from unittest.mock import patch, MagicMock
from app import utils

class TestUtils(unittest.TestCase):

    # Test la connexion a AzureChatOpenAI et AzureOpenAIEmbeddings
    @patch('app.utils.AzureChatOpenAI')
    @patch('app.utils.AzureOpenAIEmbeddings')
    def test_initialize_llm_and_embedding(self, mock_embeddings, mock_llm):
        mock_llm_instance = MagicMock()
        mock_embeddings_instance = MagicMock()
        mock_llm.return_value = mock_llm_instance
        mock_embeddings.return_value = mock_embeddings_instance

        llm, embedding = utils.initialize_llm_and_embedding()

        self.assertEqual(llm, mock_llm_instance)
        self.assertEqual(embedding, mock_embeddings_instance)
        mock_llm.assert_called_once()
        mock_embeddings.assert_called_once()

    # Test le chargement de WikipediaLoader
    # Test sur la division du document par RecursiveCharacterTextSplitter
    # Test le retriever retourne par Chroma
    @patch('app.utils.WikipediaLoader')
    @patch('app.utils.RecursiveCharacterTextSplitter')
    @patch('app.utils.Chroma')
    def test_preprocess_wikipedia(self, mock_chroma, mock_text_splitter, mock_wikipedia_loader):
        mock_loader_instance = mock_wikipedia_loader.return_value
        mock_loader_instance.load.return_value = ["Mocked Documents"]

        mock_splitter_instance = mock_text_splitter.return_value
        mock_splitter_instance.split_documents.return_value = ["Split Document 1", "Split Document 2"]

        mock_chroma_instance = MagicMock()
        mock_chroma.from_documents.return_value = mock_chroma_instance
        mock_retriever = MagicMock()
        mock_chroma_instance.as_retriever.return_value = mock_retriever

        retriever = utils.preprocess_wikipedia("test_page", "Mocked Embedding")

        self.assertEqual(retriever, mock_retriever)
        mock_wikipedia_loader.assert_called_once_with(query="test_page", load_max_docs=1, doc_content_chars_max=10000)
        mock_text_splitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        mock_chroma.from_documents.assert_called_once_with(["Split Document 1", "Split Document 2"], "Mocked Embedding")

    # Test le chargement de YoutubeLoader
    # Test sur la division du document par RecursiveCharacterTextSplitter
    # Test le retriever retourne par Chroma
    @patch('app.utils.YoutubeLoader')
    @patch('app.utils.RecursiveCharacterTextSplitter')
    @patch('app.utils.Chroma')
    def test_preprocess_youtube(self, mock_chroma, mock_text_splitter, mock_youtube_loader):
        mock_loader_instance = mock_youtube_loader.return_value
        mock_loader_instance.load.return_value = [MagicMock(page_content="Mocked Transcript")]

        mock_splitter_instance = mock_text_splitter.return_value
        mock_splitter_instance.split_documents.return_value = ["Split Document 1", "Split Document 2"]

        mock_chroma_instance = MagicMock()
        mock_chroma.from_documents.return_value = mock_chroma_instance
        mock_retriever = MagicMock()
        mock_chroma_instance.as_retriever.return_value = mock_retriever

        retriever = utils.preprocess_youtube("https://youtube.com/watch?v=test_id", "Mocked Embedding")

        self.assertEqual(retriever, mock_retriever)
        mock_youtube_loader.assert_called_once_with(video_id="test_id")
        mock_text_splitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200)
        mock_chroma.from_documents.assert_called_once_with(["Split Document 1", "Split Document 2"], "Mocked Embedding")

    # Test la connexion a create_history_aware_retriever, create_stuff_documents_chain et create_retrieval_chain
    # Test les prompts créé par ChatPromptTemplate.from_messages
    # Test que l'output soit une instance de RunnableWithMessageHistory
    @patch('app.utils.create_history_aware_retriever')
    @patch('app.utils.create_stuff_documents_chain')
    @patch('app.utils.create_retrieval_chain')
    @patch('app.utils.ChatPromptTemplate.from_messages')
    def test_create_prompts_and_chains(self, mock_chat_prompt_template, mock_create_retrieval_chain, mock_create_stuff_documents_chain, mock_create_history_aware_retriever):
        mock_llm = MagicMock()
        mock_retriever = MagicMock()

        # Créer des mocks distincts pour les retours de ChatPromptTemplate.from_messages()
        mock_contextualize_q_prompt = MagicMock()
        mock_qa_prompt = MagicMock()
        mock_chat_prompt_template.side_effect = [mock_contextualize_q_prompt, mock_qa_prompt]

        mock_history_aware_retriever = MagicMock()
        mock_create_history_aware_retriever.return_value = mock_history_aware_retriever

        mock_question_answer_chain = MagicMock()
        mock_create_stuff_documents_chain.return_value = mock_question_answer_chain

        mock_rag_chain = MagicMock()
        mock_create_retrieval_chain.return_value = mock_rag_chain

        conversational_rag_chain = utils.create_prompts_and_chains(mock_llm, mock_retriever)

        mock_create_history_aware_retriever.assert_called_once_with(mock_llm, mock_retriever, mock_contextualize_q_prompt)
        mock_create_stuff_documents_chain.assert_called_once_with(mock_llm, mock_qa_prompt)
        mock_create_retrieval_chain.assert_called_once_with(mock_history_aware_retriever, mock_question_answer_chain)

        self.assertIsInstance(conversational_rag_chain, utils.RunnableWithMessageHistory)

    # Test la connexion a create_prompts_and_chains et execute_rag_chain
    # Test que get_answer_llm retourne la réponse correcte
    @patch('app.utils.create_prompts_and_chains')
    @patch('app.utils.execute_rag_chain')
    def test_get_answer_llm(self, mock_execute_rag_chain, mock_create_prompts_and_chains):
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_conversational_rag_chain = MagicMock()
        mock_create_prompts_and_chains.return_value = mock_conversational_rag_chain
        mock_execute_rag_chain.return_value = "Mocked Answer"

        answer = utils.get_answer_llm("test_user", mock_retriever, "test_query", mock_llm)

        self.assertEqual(answer, "Mocked Answer")
        mock_create_prompts_and_chains.assert_called_once_with(mock_llm, mock_retriever)
        mock_execute_rag_chain.assert_called_once_with(mock_conversational_rag_chain, "test_user", "test_query")

if __name__ == '__main__':
    unittest.main()
