
from api_interface import rest_api_call

if __name__ == '__main__':

    question = rest_api_call("What was the revenue for Germany last year")

    print("The result is:")
    print(question)
