from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def testResponse():
    return {"message": "Hello World"}