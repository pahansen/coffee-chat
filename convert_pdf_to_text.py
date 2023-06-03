# open a pdf file and convert it to text. store the text in a string variable.

import pypdf
import os

# open the pdf file
pdfFileObj = open('/workspaces/search-embedding-example/anleitung.pdf', 'rb')
# create a pdf reader object
pdfReader = pypdf.PdfReader(pdfFileObj)
# get the number of pages in pdf file
num_pages = len(pdfReader.pages)
# initialize a count for the number of pages
count = 0
# extract text from each page on the file
while count < num_pages:
    pageObj = pdfReader.pages[count]
    count += 1
    text = pageObj.extract_text()
    # save the text for each page in a txt file
    with open(f'articles/page_{count}.txt', 'w') as f:
        f.write(text)
