import base64

def mermaid_url(graph: str) -> str:
    """Read a mermaid graph
    and return the correspondind image url
    """
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    url = "https://mermaid.ink/img/" + base64_string
    return url

tools_list = [
    {
        "type": "function",
        "function": {
            "name": "mermaid_url",
            "description": "Get the generated image's url from a mermaid graph. Call this whenever you need to display a mermaid graph",
            "parameters": {
                "type": "object",
                "properties": {
                    "graph": {
                        "type": "string",
                        "description": "The mermaid graph",
                    },
                },
                "required": ["graph"],
                "additionalProperties": False,
            },
        }
    }
]