import tornado
import tornado.web
import re

from Classifier.document_classifier import DocumentClassifier

class TextClassifier(tornado.web.RequestHandler):

    def post(self):
        # self.write(self.request)
        # print(self.request)
        # print(self.request.body)

        doc_classifier = DocumentClassifier()
        output_class = doc_classifier.classify(self.request.body)
        #print(type(output_class))
        #print(output_class[0], output_class.shape)

        #dic = {'Class': output_class[0]}
        self.write(output_class)
        



application = tornado.web.Application([
    (r"/document-classifier", TextClassifier)
])
print("above main")
def main():
    print("hello")
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
    print("Server is up !")

main()
