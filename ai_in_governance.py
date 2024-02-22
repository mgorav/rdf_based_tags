import os
from io import BytesIO
from sklearn.decomposition import PCA

import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from annoy import AnnoyIndex
import networkx as nx
from PIL import Image
from sklearn.decomposition import PCA

sns.set_theme(style='darkgrid')

# Title
st.title("Nike Data Catalog & Analytics")

import streamlit as st

# Add custom CSS to change the Streamlit theme to light
st.markdown(
    """
    <style>
    body {
        color: black;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add Nike logo with modified color pattern
st.image("https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg", width=200, output_format="PNG",
         caption="Nike")

# Load data
catalog = [
    {
        "query": "What are GS1 Standards?",
        "data_product": "GS1Standards",
        "domain": "Supply Chain",
        "ownership": "Supply Chain Management Team",
        "sourceOfTruth": "GS1",
        "description": "GS1 Standards provide a common language for identifying, capturing, and sharing supply chain data, ensuring global consistency across products, locations, and logistic units.",
        "data": {
            'Standards': ['GTIN', 'SSCC', 'GLN', 'GS1 XML'],
            'Description': [
                'Global Trade Item Number: Unique identification of trade items.',
                'Serial Shipping Container Code: Identifying logistics units.',
                'Global Location Number: Identifying locations.',
                'GS1 XML: XML-based standards for business messaging.'
            ]
        }
    },
    {
        "query": "Can you describe EDI standards?",
        "data_product": "EDIStandards",
        "domain": "Supply Chain",
        "ownership": "IT Department",
        "sourceOfTruth": "ANSI X12 and EDIFACT Documentation",
        "description": "EDI standards facilitate electronic business document exchange, including purchase orders and invoices, enhancing efficiency and reducing errors in supply chain communications.",
        "data": {
            'Standards': ['ANSI X12', 'EDIFACT'],
            'Description': [
                'ANSI X12: EDI standard used in North America.',
                'EDIFACT: International EDI standard for cross-border transactions.'
            ]
        }
    },
    {
        "query": "What is VICS?",
        "data_product": "VICSStandards",
        "domain": "Supply Chain",
        "ownership": "Supply Chain Standards Committee",
        "sourceOfTruth": "VICS Documentation",
        "description": "VICS provides guidelines for EDI implementation and catalog item synchronization, improving data accuracy and synchronization across the retail supply chain.",
        "data": {
            'Standards': ['VICS EDI guidelines', 'VICS Catalogue Item Synchronization'],
            'Description': [
                'EDI guidelines for retail supply chain data synchronization.',
                'Standards for product data synchronization between retailers and suppliers.'
            ]
        }
    },
    {
        "query": "Explain NRTS.",
        "data_product": "NRTSStandards",
        "domain": "Supply Chain",
        "ownership": "National Retail Federation",
        "sourceOfTruth": "NRTS Documentation",
        "description": "NRTS offers XML-based standards for exchanging product data and inventory levels, facilitating efficient communication between retailers and their partners.",
        "data": {
            'Standards': ['Nationwide Retail Transfer Standard'],
            'Description': [
                'XML-based standards for product data and inventory level exchange.'
            ]
        }
    },
    {
        "query": "How are ASN and RA managed?",
        "data_product": "ASNandRA",
        "domain": "Logistics",
        "ownership": "Logistics Department",
        "sourceOfTruth": "Supply Chain Operations Reference",
        "description": "Advance Ship Notices and Receiving Advice documents streamline receiving and inventory management by providing detailed information on shipments and received goods.",
        "data": {
            'Documents': ['Advance Ship Notice (ASN)', 'Receiving Advice (RA)'],
            'Description': [
                'ASN: Provides detailed info about an upcoming shipment.',
                'RA: Acknowledges the receipt of goods and reconciles discrepancies.'
            ]
        }
    },
    {
        "query": "Describe POS data exchange standards.",
        "data_product": "POSDataExchange",
        "domain": "Retail",
        "ownership": "Retail Operations Team",
        "sourceOfTruth": "Retail Management Systems",
        "description": "Standards for sharing point-of-sale data to enable real-time sales tracking and inventory management, fostering collaborative planning and replenishment.",
        "data": {
            'Standards': ['EDI', 'GS1 XML'],
            'Benefits': [
                'Enables real-time sales tracking.',
                'Facilitates inventory management.'
            ]
        }
    },
    {
        "query": "What is the customer schema?",
        "data_product": "Customer360",
        "domain": "Customer",
        "ownership": "Customer Relations Team",
        "sourceOfTruth": "CRM System",
        "description": "Detailed customer profiles including purchase history and engagement metrics, vital for personalized marketing and customer support.",
        "data": {
            'Table': ['CustomerInfo', 'EngagementScores'],
            'Columns': [['CustomerID', 'Name', 'Email'], ['CustomerID', 'Score']],
            'Privacy': [['PII', 'PII', 'PII'], ['-', 'Confidential']],
            'Tags': [['Core', 'Core', 'Core'], ['Metric', 'Metric']]
        }
    },
    {
        "query": "Can you describe the product schema?",
        "data_product": "ProductCatalog",
        "domain": "Inventory",
        "ownership": "Product Management Team",
        "sourceOfTruth": "Product Database",
        "description": "Comprehensive product listings, including stock levels and categories, enabling inventory management and sales strategy planning.",
        "data": {
            'Table': ['ProductInfo', 'StockLevels'],
            'Columns': [['ProductID', 'Name', 'Category'], ['ProductID', 'StockAvailable']],
            'Privacy': [['-', '-', '-'], ['-', '-']],
            'Tags': [['Core', 'Core', 'Attribute'], ['Metric', 'Metric']]
        }
    },
    {
        "query": "Show the sales lineage.",
        "data_product": "SalesData",
        "domain": "Sales",
        "ownership": "Sales Analysis Team",
        "sourceOfTruth": "Sales Database",
        "description": "Tracks the journey of sales data from order capture to final analysis, critical for understanding sales trends and customer preferences.",
        "data": {
            'source': ['OrderInfo', 'CustomerInfo'],
            'target': ['SalesAnalysis', 'CustomerSegmentation'],
            'Columns': ['OrderID, ProductID, CustomerID', 'CustomerID, Segment'],
            'Table and Columns': [
                'OrderInfo -> SalesAnalysis: OrderID, ProductID, CustomerID',
                'CustomerInfo -> CustomerSegmentation: CustomerID, Segment'
            ]
        }
    },
    {
        "query": "What is the inventory readiness?",
        "data_product": "InventoryManagement",
        "domain": "Inventory",
        "ownership": "Inventory Control Team",
        "sourceOfTruth": "Warehouse Management System",
        "description": "Assesses inventory accuracy and availability, ensuring stock levels meet demand without overstocking, essential for efficient supply chain management.",
        "data": {
            'Metric': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness'],
            'Score': [98, 99, 97, 95]
        }
    },
    {
        "query": "What are the customer usage patterns?",
        "data_product": "Customer360",
        "domain": "Customer",
        "ownership": "Customer Insights Team",
        "sourceOfTruth": "Usage Tracking System",
        "description": "Analyzes customer interaction and purchasing patterns over time to refine product offerings and marketing strategies.",
        "data": {
            'Month': ['January', 'February', 'March', 'April'],
            'Usage': [220, 230, 210, 250]
        }
    },
    {
        "query": "Explain the POS data exchange standards.",
        "data_product": "POSDataExchange",
        "domain": "Sales",
        "ownership": "Sales Analysis Team",
        "sourceOfTruth": "Retail POS Systems",
        "description": "Facilitates the seamless exchange of sales data between retail points of sale and corporate systems, enhancing real-time sales tracking and inventory management.",
        "data": {
            'Standards': ['EDI', 'GS1 XML'],
            'Documents': ['Invoices', 'Purchase Orders'],
            'Frequency': ['Daily', 'Weekly'],
            'Partners': ['Retailers', 'Suppliers']
        }
    },
    {
        "query": "How is shipment tracking managed in the logistics domain?",
        "data_product": "ShipmentTracking",
        "domain": "Logistics",
        "ownership": "Logistics Team",
        "sourceOfTruth": "Logistics Systems",
        "description": "Monitors the real-time status of shipments across the supply chain, ensuring timely delivery and minimizing disruptions.",
        "data": {
            'Standards': ['GS1 Standards', 'EDI'],
            'Documents': ['Advance Ship Notices (ASN)', 'Receiving Advice (RA)'],
            'Frequency': ['Real-time', 'Periodic'],
            'Partners': ['Carriers', 'Warehouses']
        },

    }
]

# Define questions with more detailed prompts
questions = [
    "What is the customer schema?",
    "Can you describe the product schema?",
    "Show the sales lineage.",
    "What is the inventory readiness?",
    "What are the customer usage patterns?",
    "Explain the POS data exchange standards.",
    "How is shipment tracking managed in the logistics domain?",
    "What are GS1 Standards?",
    "Can you describe EDI standards?",
    "What is VICS?",
    "Explain NRTS.",
    "How are ASN and RA managed?",
    "Describe POS data exchange standards.",
    "Outline the data privacy policies for customer information.",
    "What are the security protocols for product data?",
    "Describe the governance framework for sales data lineage.",
    "How is data quality in inventory management ensured?",
    "Explain the consent management process for customer data usage.",
    "What compliance standards are followed for supply chain data exchanges?",
    "Describe the process for third-party data sharing and agreements.",
    "How is sensitive data identified and protected in customer profiles?",
    "Detail the audit processes for data governance compliance.",
    "Explain the role and responsibilities of the data governance committee.",
    "What are the procedures for data breach response and notification?",
    "How is anonymization applied to sales and customer data for analysis?"
]

# Define the path to your data catalog image
data_catalog_image_path = "data_catalog.png"

# Define catalog structure text
catalog_structure_text = """
The data catalog consists of various domains, each with specific data products owned by designated teams and linked to their respective sources of truth. 
- The **Customer Domain** includes 'Customer360' and 'CustomerUsagePattern', owned by the Customer Relations Team and Customer Insights Team, respectively, with CRM and Usage Tracking Systems as sources of truth.
- The **Inventory Domain** covers 'ProductCatalog' and 'InventoryManagement', managed by the Product Management Team and Inventory Control Team, with the Product Database and Warehouse Management System as sources of truth.
- Under the **Sales Domain**, 'SalesData' and 'POSDataExchange' are analyzed by the Sales Analysis Team and rely on the Sales Database and Retail POS Systems.
- The **Logistics Domain** offers 'ShipmentTracking', owned by the Logistics Team with the Logistics Systems as the source of truth.
"""

# Pretrained model selection
model_options = {
    "MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "DistilBERT": "distilbert-base-uncased",
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base"
}

selected_model = st.sidebar.selectbox("Select Pretrained Model", list(model_options.keys()), index=0)


# Load tokenizer and model
@st.cache_data
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_options[model_name])
    model = AutoModel.from_pretrained(model_options[model_name])
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer(selected_model)


def display_data_catalog():
    # Function to display the data catalog image and explanation
    st.image(data_catalog_image_path, caption='Data Catalog Structure')
    st.markdown(catalog_structure_text, unsafe_allow_html=True)


@st.cache_data
def create_embeddings(catalog):
    # Extract texts from catalog
    texts = [item["query"] for item in catalog]

    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Use a specific layer's output instead of pooler_output
    embeddings = model_output.last_hidden_state
    # Ensure the dimensionality is reduced to 384 explicitly
    # This part might need adjustment based on your specific dimensionality reduction method
    reduced_embeddings = torch.mean(embeddings, dim=1)
    if reduced_embeddings.shape[1] != 384:
        # Add your dimensionality reduction code here to adjust to 384 if necessary
        pass
    reduced_embeddings = torch.nn.functional.normalize(reduced_embeddings)
    return reduced_embeddings.detach().cpu().numpy()



@st.cache_data(hash_funcs={tokenizer.__class__: lambda _: None})
def retrieve_data_for_query(query):
    for item in catalog:
        if item["query"].lower() in query.lower():
            data = item['data']
            if 'Table' in data and 'Columns' in data:
                schema_df = pd.DataFrame({
                    'Table': sum([[table] * len(cols) for table, cols in zip(data['Table'], data['Columns'])], []),
                    'Column': sum(data['Columns'], []),
                })
                return schema_df, None
            else:
                df = pd.DataFrame(data)
                additional_df = None
                if 'Table and Columns' in data:
                    additional_df = pd.DataFrame({'Table and Columns': data['Table and Columns']})
                    df.drop(columns=['Table and Columns'], inplace=True)
                return df, additional_df
    return pd.DataFrame(), None


def generate_dynamic_chart(df, query):
    if "usage pattern" in query.lower():
        plt.figure(figsize=(10, 6))
        plt.plot(df['Month'], df['Usage'], marker='o', color='skyblue')
        plt.title("Customer Usage Pattern", fontsize=16)
        plt.xlabel("Month", fontsize=14)
        plt.ylabel("Usage", fontsize=14)
        plt.grid(True)
        st.pyplot(plt)
    elif "inventory readiness" in query.lower():
        plt.figure(figsize=(8, 6))
        plt.bar(df['Metric'], df['Score'], color='lightgreen')
        plt.title("Inventory Readiness Index", fontsize=16)
        plt.xlabel("Metric", fontsize=14)
        plt.ylabel("Score", fontsize=14)
        plt.xticks(rotation=45)
        st.pyplot(plt)
    else:
        G = nx.DiGraph()
        pos = {}  # Initialize `pos` to ensure it's available
        if "schema" in query.lower():
            for _, row in df.iterrows():
                G.add_node(row['Table'], type='Table', color='skyblue', label=row['Table'])
                G.add_node(row['Column'], type='Column', color='lightgreen', label=row['Column'])
                G.add_edge(row['Table'], row['Column'])
            pos = nx.spring_layout(G, k=0.15, iterations=20)
        elif "lineage" in query.lower():
            for idx, row in df.iterrows():
                G.add_node(row['source'], type='source', color='skyblue', label=row['source'])
                G.add_node(row['target'], type='target', color='orange', label=row['target'])
                G.add_edge(row['source'], row['target'], label=row.get('Columns', ''))
            pos = nx.spring_layout(G, k=0.30, iterations=40)

        if pos:  # Check if `pos` is not empty
            plt.figure(figsize=(12, 8))
            nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=2000,
                    node_color=[data['color'] for _, data in G.nodes(data=True)], font_size=10, font_weight='bold',
                    edge_color='gray', width=2, arrowstyle='->', arrowsize=10)
            if "lineage" in query.lower():
                edge_labels = nx.get_edge_attributes(G, 'label')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
            plt.title("Data Product Schema & Lineage Visualization", fontsize=16)
            plt.axis('off')
            st.pyplot(plt)


def display_data_ownership():
    # New function to display data product ownership information
    ownership_data = [{"Data Product": item["data_product"], "Ownership": item["ownership"]} for item in catalog]
    df_ownership = pd.DataFrame(ownership_data)
    sns.barplot(data=df_ownership, x="Data Product", y="Ownership")
    plt.xticks(rotation=45)
    plt.title("Ownership of Data Products")
    st.pyplot(plt)


def main():
    # Assuming the initialization of embeddings and Annoy index is done elsewhere or included here as commented
    embeddings = create_embeddings(catalog)  # Ensure this is called appropriately
    annoy_index = AnnoyIndex(embeddings.shape[1], 'angular')
    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)
    annoy_index.build(10)  # This builds the Annoy index with the catalog embeddings

    st.sidebar.title("Select a Question or View:")
    options = ["Data Catalog"] + questions + ["View Data Product Ownership"]
    selected_option = st.sidebar.selectbox("Options", options)

    # Initialize the summarization pipeline here to avoid reloading for each execution
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    @st.cache_data
    def summarize_text(text):
        """
        Summarizes the provided text using the initialized summarization pipeline.
        """
        # Perform summarization
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
        # Return the summarized text
        return summary[0]['summary_text']

    if selected_option == "Data Catalog":
        display_data_catalog()
    elif selected_option == "View Data Product Ownership":
        display_data_ownership()
    else:
        st.subheader(f"Question: {selected_option}")

        # Adjusted to handle a single selected question correctly
        # Wrapping the selected question in a dict to match the expected input format for create_embeddings
        question_embedding = create_embeddings([{"query": selected_option}])[0]

        # Use Annoy to find the nearest query embedding
        nearest_ids = annoy_index.get_nns_by_vector(question_embedding, 1, include_distances=False)
        if nearest_ids:
            nearest_id = nearest_ids[0]
            matched_query = catalog[nearest_id]['query']
            st.write(f"Matching query in catalog: {matched_query}")

            # Retrieve and display data based on the matched query
            df, additional_df = retrieve_data_for_query(matched_query)
            summary_text = ""
            if not df.empty:
                st.write(df)
                summary_text += df.to_string()  # Convert DataFrame to string for summarization
                if additional_df is not None:
                    st.write(additional_df)
                    summary_text += "\n" + additional_df.to_string()
                generate_dynamic_chart(df, selected_option)
            else:
                st.write("No data available for this query.")
                summary_text = "No data available for this query."

            # Summarize the displayed information if any
            if summary_text:
                summarized_info = summarize_text(summary_text)
                st.subheader("Summary of Understanding")
                st.write(summarized_info)
        else:
            st.write("No matching query found in the catalog.")


if __name__ == "__main__":
    # Ensure the embeddings and Annoy index are created only when running the script
    embeddings = create_embeddings(catalog)
    annoy_index = AnnoyIndex(embeddings.shape[1], 'angular')

    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)

    annoy_index.build(10)
    main()
