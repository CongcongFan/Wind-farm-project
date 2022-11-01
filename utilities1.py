from html.parser import HTMLParser
import re
import os

def strip_tags(html):
    if isinstance(html,list):
        html = ''.join(html)
    html = html.strip()
    # html = html.strip("\n")
    result = []
    parser = HTMLParser()
    parser.handle_data = result.append
    parser.feed(html)
    parser.close()
    result = ''.join(result)
    # result = result.replace('\n','')
    return result

def extract_chinese(txt, with_punctuation = True):
    if with_punctuation:
        pattern = re.compile(u'[\u4e00-\u9fa5-\n\，\。]')
    else:
        pattern = re.compile(u'[\u4e00-\u9fa5-\n]')
    return ''.join(pattern.findall(txt))

def write_file(fname, str, path = '/Users/cong/iCloud/文稿2/K/小说'):
    path = os.path.expanduser(path)
    path = os.path.join(path, fname)
    with open(path,'a') as o:
        o.write(str)

###### PDF eqn ###########
from pdfreader import SimplePDFViewer, PageDoesNotExist

fd = open('/Users/cong/iCloud/Desktop/Monash/MEC4408/eqn.pdf', "rb")
viewer = SimplePDFViewer(fd)

plain_text = ""
pdf_markdown = ""
images = []
try:
    while True:
        viewer.render()
        pdf_markdown += viewer.canvas.text_content
        plain_text += "".join(viewer.canvas.strings)
        images.extend(viewer.canvas.inline_images)
        images.extend(viewer.canvas.images.values())
        viewer.next()
except PageDoesNotExist:
    pass
###### PDF eqn ################# PDF eqn ################# PDF eqn ###########