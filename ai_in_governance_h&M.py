import os
from io import BytesIO

import seaborn as sns
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tokenizers import tokenizers
from transformers import AutoTokenizer, AutoModel
import torch
from annoy import AnnoyIndex
import networkx as nx
from PIL import Image

sns.set_theme(style='darkgrid')

# Title
st.title("H&M Data Catalog & Analytics")

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
st.image("HM.png", width=200, output_format="PNG")



# Load data
catalog = [
    {
        "query": "customer schema",
        "data_product": "Customer360",
        "domain": "Customer",
        "ownership": "Customer Relations Team",
        "sourceOfTruth": "CRM System",
        "data": {
            'Table': ['CustomerInfo', 'EngagementScores'],
            'Columns': [['CustomerID', 'Name', 'Email'], ['CustomerID', 'Score']],
            'Privacy': [['PII', 'PII', 'PII'], ['-', 'Confidential']],
            'Tags': [['Core', 'Core', 'Core'], ['Metric', 'Metric']]
        }
    },
    {
        "query": "product schema",
        "data_product": "ProductCatalog",
        "domain": "Inventory",
        "ownership": "Product Management Team",
        "sourceOfTruth": "Product Database",
        "data": {
            'Table': ['ProductInfo', 'StockLevels'],
            'Columns': [['ProductID', 'Name', 'Category'], ['ProductID', 'StockAvailable']],
            'Privacy': [['-', '-', '-'], ['-', '-']],
            'Tags': [['Core', 'Core', 'Attribute'], ['Metric', 'Metric']]
        }
    },
    {
        "query": "sales lineage",
        "data_product": "SalesData",
        "domain": "Sales",
        "ownership": "Sales Analysis Team",
        "sourceOfTruth": "Sales Database",
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
        "query": "inventory readiness",
        "data_product": "InventoryManagement",
        "domain": "Inventory",
        "ownership": "Inventory Control Team",
        "sourceOfTruth": "Warehouse Management System",
        "data": {
            'Metric': ['Completeness', 'Accuracy', 'Consistency', 'Timeliness'],
            'Score': [98, 99, 97, 95]
        }
    },
    {
        "query": "customer usage pattern",
        "data_product": "Customer360",
        "domain": "Customer",
        "ownership": "Customer Insights Team",
        "sourceOfTruth": "Usage Tracking System",
        "data": {
            'Month': ['January', 'February', 'March', 'April'],
            'Usage': [220, 230, 210, 250]
        }
    },
    {
        "query": "POS data exchange",
        "data_product": "POSDataExchange",
        "domain": "Sales",
        "ownership": "Sales Analysis Team",
        "sourceOfTruth": "Retail POS Systems",
        "data": {
            'Standards': ['EDI', 'GS1 XML'],
            'Documents': ['Invoices', 'Purchase Orders'],
            'Frequency': ['Daily', 'Weekly'],
            'Partners': ['Retailers', 'Suppliers']
        }
    },
    {
        "query": "shipment tracking",
        "data_product": "ShipmentTracking",
        "domain": "Logistics",
        "ownership": "Logistics Team",
        "sourceOfTruth": "Logistics Systems",
        "data": {
            'Standards': ['GS1 Standards', 'EDI'],
            'Documents': ['Advance Ship Notices (ASN)', 'Receiving Advice (RA)'],
            'Frequency': ['Real-time', 'Periodic'],
            'Partners': ['Carriers', 'Warehouses']
        }
    }
]

questions = [
    "What is the customer schema?",
    "Can you describe the product schema?",
    "Show the sales lineage.",
    "What is the inventory readiness?",
    "What are the customer usage patterns?",
    # Additional Questions
    "Explain the POS data exchange standards.",
    "How is shipment tracking managed in the logistics domain?",
    # Business Angle Questions
    "How do supply chain exchange standards enhance collaboration between retailers and suppliers?",
    "What benefits do retailers derive from implementing EDI standards like ANSI X12?",
    "How does inventory management contribute to optimizing supply chain operations?",
    "What role does customer usage pattern analysis play in improving retail strategies?",
    "Discuss the importance of accurate sales lineage for sales analysis and customer segmentation."
]

@st.cache_data(hash_funcs={AutoTokenizer: lambda _: None})
def create_embeddings(catalog):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Extract texts from catalog
    texts = [item["query"] for item in catalog]

    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.pooler_output
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings.detach().cpu().numpy()

@st.cache_data(hash_funcs={tokenizers.Tokenizer: lambda _: None, tokenizers.AddedToken: lambda _: None})
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
    st.sidebar.title("Select a Question or View:")
    options = [""] + questions + ["View Data Product Ownership"]
    selected_option = st.sidebar.selectbox("Options", options)

    if selected_option:
        if selected_option == "View Data Product Ownership":
            display_data_ownership()
        else:
            st.subheader(f"Question: {selected_option}")
            question_index = None
            for i, item in enumerate(catalog):
                if item["query"].lower() in selected_option.lower():
                    question_index = i
                    break

            if question_index is not None:
                question_embedding = create_embeddings([catalog[question_index]])[0]
                nearest_ids = annoy_index.get_nns_by_vector(question_embedding, 1, include_distances=False)
                for nearest_id in nearest_ids:
                    matched_query = catalog[nearest_id]['query']
                    st.write(f"Matching query in catalog: {matched_query}")
                    df, additional_df = retrieve_data_for_query(matched_query)
                    if not df.empty:
                        st.write(df)
                        if additional_df is not None:
                            st.write(additional_df)
                        generate_dynamic_chart(df, selected_option)
                    else:
                        st.write("No data available for this query.")
            else:
                st.write("Selected question not found in the list.")

if __name__ == "__main__":
    embeddings = create_embeddings(catalog)
    annoy_index = AnnoyIndex(embeddings.shape[1], 'angular')

    for i, embedding in enumerate(embeddings):
        annoy_index.add_item(i, embedding)

    annoy_index.build(10)
    main()
