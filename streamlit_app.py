import streamlit as st
from openai import OpenAI
from tools import mermaid_url, tools_list
import json
import pandas as pd

# Load corpus
CORPUS_PATH = './corpus/'
CORPUS_FILE = CORPUS_PATH + "2025-02-25 SOCIOSCOPE CORPUS.json"

with open(CORPUS_FILE, 'r') as f:
    corpus = json.load(f)

print(f"Loaded corpus: {len(corpus)} records.")

# Transform as dataframe
corpus_df = pd.DataFrame(corpus)
metadatas_df = pd.json_normalize(corpus_df['metadata'])
corpus_df = pd.concat([metadatas_df, corpus_df], axis=1)
corpus_df = corpus_df.drop(columns=['metadata'])

projects = corpus_df['PROJECT'].unique()
print(f"Found {len(projects)} projects.")

# Show title and description.
st.title("📄 Socioscope Corpus RAG")
st.write(
    "Choose a project below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")

else:
    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    #uploaded_file = st.file_uploader(
    #    "Upload a document (.txt or .md)", type=("txt", "md")
    #)

    # Choose a project from an automatic list
    options = st.multiselect('Select available projects:', projects)

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the selected projects",
        placeholder="Can you give me a short summary?",
        disabled=not (len(options)<1),
    )

    if options and question:
        filter = corpus_df['PROJECT'].isin(options)
        documents = corpus_df[filter][['PROJECT', 'content']]
        print(f"Querying {len(documents)} documents.")

        # Process the chosen projects and question.
        document = '\n\n'.join(documents['content'])

        stream_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful academic research assistant." 
                    # "Encode your response in a mermaid graph and use the 'mermaid_url' tool when user ask for a graph"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Here's a document about {len(options)} different projects:" 
                    f"{document} \n\n"
                    f"---\n\n {question}"
                ),
            }
        ]

        # Generate an answer using the OpenAI API.
        stream_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=stream_messages,
            # tools=tools_list,
            stream=True,
        )
        response = st.write_stream(stream_response)
        graph_messages = [
            {
                "role": "system",
                "content": (
                    "1. Encode the user response in a mermaid graph"
                    "2. Use the 'mermaid_url' tool"
                )
            },
            {
                "role": "user",
                "content": (
                    f"{response}"
                ),
            }
        ]

        graph_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=graph_messages,
            tools=tools_list,
        )

        if graph_response.choices:
            tool_call = graph_response.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)
            graph = arguments.get('graph')
            url = mermaid_url(graph)
            st.write("\n\n# Graphical representation")
            st.image(url+'?type=webp', use_container_width=True)
