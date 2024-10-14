import logging
from backend.worker import parse_dataset, embedding_doc, retrieval_and_generate

def main():
    "# Step 1: Parse the dataset
    logging.info("Starting dataset parsing...")
    dataset_name = 'sustainability'  # You can switch between 'sustainability' and 'christmas'
    parsed_doc = parse_dataset(dataset_name)

    if 'error' in parsed_doc:
        logging.error(f"Failed to parse dataset: {parsed_doc['error']}")
        return
    
    # Step 2: Embed the document in the vector store
    logging.info("Embedding document into vector store...")
    try:
        embedding_doc(parsed_doc)
    except Exception as e:
        logging.error(f"Embedding failed: {e}")
        return
    
    # Step 3: Perform a sample query
    logging.info("Performing a sample query...")"
    query = "What are consumers' attitudes toward sustainability in the UK?"
    
    try:
        response = retrieval_and_generate(query)
        logging.info(f"Query response: {response}")
    except Exception as e:
        logging.error(f"Query failed: {e}")
        return
    
    logging.info("Test completed successfully.")

if __name__ == '__main__':
    main()
