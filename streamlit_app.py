import streamlit as st
from openai import OpenAI
from tools import mermaid_url, tools_list
import vectordb, rag
import json
import pandas as pd

# Show title and description.
st.title("📄 Socioscope Corpus RAG")
st.write(
    "Ask a question about the socioscope corpus – GPT4 will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")

else:
    # Load corpus
    CORPUS_PATH = './corpus/'
    CORPUS_VDB = CORPUS_PATH + "2025-02-28 SOCIOSCOPE.vdb"
    db = vectordb.load_vectorstore(CORPUS_VDB, openai_api_key)
    graph = rag.create_llm_graph(db, openai_api_key)

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the socioscope projects",
        placeholder="What is EDEN?",
        # disabled=(len(options)<1),
    )

    if question:
        result = graph.invoke({"question":question})
        response = result['answer']

        if response:
            # Answer
            st.write("# Answer")
            st.write(response.answer)

            # Citations
            st.write("# Citations")
            for idx, row in enumerate(response.citations):
                st.write(f"**[{idx}] {result['context'][row.source_id].metadata['FILE']}**")
                st.write(f'*"{row.quote}"*')

        # Graphical representation
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
                    f"{response.answer}"
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
