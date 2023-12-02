import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.dbmodels import RequestHistory
from app.database import SessionLocal
from app.model import predict

# import model

app = FastAPI(
    title="CodeSummarization"
)

app.mount("/app/static", StaticFiles(directory=Path(__file__).parent.parent.absolute() / "app/static"), name="static")
# Jinja2 templates configuration
templates = Jinja2Templates(directory="src/app/templates")


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def process_form(request: Request, code: str = Form(...)):
    result_summary = predict(code)

    # Add request to history
    db = SessionLocal()
    db_request = RequestHistory(code=code, result_summary=result_summary)
    db.add(db_request)
    db.commit()
    db.refresh(db_request)
    db.close()

    return templates.TemplateResponse("result.html",
                                      {"request": request, "code": code, "result_summary": result_summary})


@app.get("/history", response_class=HTMLResponse)
async def get_history(request: Request):
    db = SessionLocal()
    request_history = db.query(RequestHistory).all()
    db.close()

    return templates.TemplateResponse("history.html", {"request": request, "request_history": request_history})


def start():
    uvicorn.run("app.main:app", reload=True)
