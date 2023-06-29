# Search embedding example - Coffee Chat
In this example, you find a simple chatbot which answers questions about a Tchibo coffee machine.
All questions are anwered based on the manual for this cofee machine https://www.tchibo.de/newmedia/document/1274939cdf4d54ad/anleitung.pdf.
The bot will only use the information provided in the document and drop a little custom advertisement whenever it feels the user is asking for coffee recommendations.

The file `tchibo_search.ipynb` contains a custom example how to use the OpenAI API to build a chain of commands to prep and retrieve documents and answer questions based on the content.

In `main.py` the whole chain was implemented as a command line tool using the Conversational Retrieval QA chain from Langchain. Obviously, this is the prefered approach because it speeds up development due to a lot of abstractions already implemented to achieve the same result.

