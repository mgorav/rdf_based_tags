import json
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from annoy import AnnoyIndex
from langchain_core.outputs import LLMResult
from transformers import AutoTokenizer, AutoModel, pipeline
from langchain_community.llms import OpenAI
from data_catalog import catalog, personas, model_options, catalog_structure_text

openai_api_key = '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'
llm = OpenAI(api_key=openai_api_key)

sns.set_theme(style='darkgrid')

# Title
st.title("Nike Data Catalog & Analytics")

# Add Nike logo with modified color pattern
st.image("https://upload.wikimedia.org/wikipedia/commons/a/a6/Logo_NIKE.svg", width=200, output_format="PNG",
         caption="Nike")

# Define the path to your data catalog image
data_catalog_image_path = "data_catalog.png"

selected_model = st.sidebar.selectbox("Select Pretrained Model", list(model_options.keys()), index=0)


# Load tokenizer and model
@st.cache_data
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_options[model_name])
    model = AutoModel.from_pretrained(model_options[model_name])
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer(selected_model)

selected_persona = st.sidebar.selectbox("Select Persona", list(personas.keys()), index=0)

# Load the selected persona's questions
questions = personas[selected_persona]


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
    # summarizer = pipeline("summarization", model="facebook/bart")
    summarizer = pipeline("summarization", model="t5-small")

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
    # New Data Steward Workflow
    elif selected_persona == "Data Steward":
        st.sidebar.subheader('Data Steward Workflow')
        selected_workflow = st.sidebar.selectbox("Select Workflow", ["", "Collibra"])

        if selected_workflow == "Collibra":
            st.empty()
            st.subheader('Collibra Data Products')
            # Display table of data products
            products = [item["data_product"] for item in catalog]
            selected_products = st.multiselect("Select Data Products", products)
            if selected_products:
                selected_catalog = [item for item in catalog if item["data_product"] in selected_products]

                # Initialize a list to keep track of checkbox states
                checkbox_states = []

                # Create a dataframe for selected products
                selected_catalog_df = pd.DataFrame(selected_catalog)

                # Create a new column in the dataframe to store the checkbox widgets
                selected_catalog_df['Select'] = False

                # Iterate over the dataframe and display checkboxes with data
                for index, row in selected_catalog_df.iterrows():
                    cols = st.columns([1, 5, 10])

                    # Display checkbox in the first column
                    with cols[0]:
                        selected_catalog_df.at[index, 'Select'] = st.checkbox('', value=row['Select'], key=str(index))

                    # Display product name and description in the second column
                    with cols[1]:
                        st.write(f"**Product:** {row['data_product']}")
                        st.write(f"**Description:** {row['description']}")

                    # Display JSON data in the third column using a tree view
                    with cols[2]:
                        # st.json(row['data'])
                        st.dataframe(row['data'])

                # When the submit button is pressed
                if st.button("Submit Data Product Registrations"):
                    # Iterate through the dataframe and check the state of each checkbox
                    for index, row in selected_catalog_df.iterrows():
                        if row['Select']:  # If the checkbox is selected
                            product_name = row['data_product']
                            submit_to_collibra(product_name, row['data'], st)
                            # st.write(f"Data product '{product}' is being submitted to Collibra.")

                        else:
                            st.write("Please select at least one data product.")

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
                    summary_text += "\n" + additional_df.to_string()  # Append additional DataFrame to string
                generate_dynamic_chart(df, matched_query)
            else:
                st.write("No data found for the selected query.")

            # Perform summarization if data and additional data are available
            if summary_text:
                st.subheader("Summary:")
                summary_result = summarize_text(summary_text)
                st.write(summary_result)
        else:
            st.write("No matching query found in the catalog.")


import logging

logger = logging.getLogger(__name__)

# def submit_to_collibra(product_name, json_payload, st):
#     st.write(f"Attempting to submit '{product_name}' to Collibra.")
#
#     if not isinstance(json_payload, str):
#         json_payload_str = json.dumps(json_payload, ensure_ascii=False)
#     else:
#         json_payload_str = json_payload
#
#     prompt = f"""
#     Given the JSON payload for '{product_name}', convert it into a detailed JSON format suitable for Collibra registration, including necessary fields like community, domain, asset, table, columns, and owner:
#     JSON Payload: {json_payload_str}
#     """
#
#     try:
#         response = llm.generate(prompts=[prompt], max_tokens=1024)
#
#         if hasattr(response, "generations") and response.generations:
#
#             generation = response.generations[0]
#
#             if hasattr(generation, "text"):
#                 generated_text = generation.text.strip()
#
#             else:
#                 print(type(generation))
#                 generated_text = json.dumps(generation[0].text.strip())
#
#             try:
#                 collibra_payload = json.loads(generated_text)
#                 # st.json(collibra_payload)
#                 # First, parse the JSON string into a Python object
#                 data = json.loads(collibra_payload)
#
#                 # Then, create a DataFrame from this object
#                 df = pd.DataFrame(data)
#
#                 # Display the DataFrame in Streamlit
#                 st.dataframe(df)
#
#             except json.JSONDecodeError:
#                 logger.error("Invalid JSON format")
#                 st.error("Invalid JSON format")
#
#         else:
#             logger.debug("No generations returned")
#             st.error("No valid response")
#
#     except Exception as e:
#         logger.exception("Submission error")
#         st.error(f"An error occurred: {e}")

import json
import pandas as pd


def create_and_display_tables(data, st, parent_key='', path=''):
    """
    Recursively create and display tables for nested dictionaries and lists with improved display.
    """
    if isinstance(data, dict):
        # Flatten the dictionary to handle nested structures with improved descriptions
        flat_data = {}
        for k, v in data.items():
            new_path = f"{path}.{k}" if path else k
            if isinstance(v, dict):
                flat_data[k] = "; ".join([f"{key}: {value}" for key, value in v.items() if not isinstance(value, (dict, list))])
                create_and_display_tables(v, st, k, new_path)
            elif isinstance(v, list):
                # Attempt to summarize list content if it's not too complex
                if all(not isinstance(i, (dict, list)) for i in v):
                    flat_data[k] = ", ".join(map(str, v))
                else:
                    flat_data[k] = "See below for details"
                    create_and_display_tables(v, st, k, new_path)
            else:
                flat_data[k] = v

        if flat_data:
            df = pd.DataFrame([flat_data])
            st.write(f"### {parent_key}", df)

    elif isinstance(data, list):
        # Process lists containing dictionaries
        if data and isinstance(data[0], dict):
            # Simplify list items for display if they are not too complex
            simplified_data = [
                {k: v if not isinstance(v, (dict, list)) else "See details below" for k, v in item.items()}
                for item in data
            ]
            df = pd.DataFrame(simplified_data)
            st.write(f"### {parent_key}", df)
            # Recursively handle complex items in the list
            for i, item in enumerate(data):
                for k, v in item.items():
                    new_path = f"{path}[{i}].{k}"
                    if isinstance(v, (dict, list)):
                        create_and_display_tables(v, st, f"{k} in {parent_key}[{i}]", new_path)
        else:
            # Directly display simple lists
            st.write(f"### {parent_key}", data)



def submit_to_collibra(product_name, json_payload, st):
    st.write(f"Attempting to submit '{product_name}' to Collibra.")

    prompt = f"""
    Create a JSON payload that aligns with Collibra's Swagger specifications for data product registration, 
    including all necessary fields like community, domain, asset, table, columns, and owner. The JSON 
    should be structured to facilitate the onboarding of the '{product_name}' data product into Collibra, 
    ensuring compliance with the following aspects based on Collibra's REST API requirements:
    
    - Community and domain creation with relevant identifiers
    - Data product asset creation including name, description, and linkage to the created community
    - Definition of tables as entities within the data product, including columns as attributes
    - Assignment of roles and responsibilities to stewards for each domain, community, and data product
    
    JSON Payload for reference: {json.dumps(json_payload, ensure_ascii=False)}
    
    Please provide the JSON in a format that can be directly used for API calls to register a data product in Collibra.
    """

    try:
        response = llm.generate(
            prompts=[prompt],
            model="gpt-3.5-turbo-instruct",
            max_tokens=1024,
            temperature=0.7,
        )

        # Assuming the correct way to access the generated text is through the 'text' attribute of the first 'Generation' object
        if response.generations and response.generations[0]:
            # Directly accessing the 'text' attribute of the first generation
            generation = response.generations[0][0]  # Assuming the first item in the list is what we need
            generated_text = generation.text.strip()  # Accessing 'text' attribute and stripping leading/trailing whitespace
        else:
            raise AttributeError("Generated text not found in the response")

        collibra_payload = _parse_llm_response(generated_text)

        # Display the parsed Collibra JSON in Streamlit
        create_and_display_tables(collibra_payload, st)

    except json.JSONDecodeError:
        logger.error("Invalid JSON format from LLM")
        st.error("Invalid JSON format from LLM")
    except Exception as e:
        logger.exception("Submission error")
        st.error(f"An error occurred: {e}")


def _parse_llm_response(generated_text):
    try:
        # Directly parsing the generated_text as it's already a string
        return json.loads(generated_text)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in parsed response from GPT")


if __name__ == "__main__":
    main()
