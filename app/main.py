import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# import model

app = FastAPI(
    title="CodeSummarization"
)

# Jinja2 templates configuration
templates = Jinja2Templates(directory="app/templates")


# initialize model

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def process_form(request: Request, code: str = Form(...)):
    result_summary = "mock code summarization"  # Replace with your actual prediction function

    return templates.TemplateResponse("result.html",
                                      {"request": request, "code": code, "result_summary": result_summary})


def start():
    uvicorn.run("app.main:app", reload=True)
