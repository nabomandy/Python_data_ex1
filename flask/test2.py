from re import template
import tensorflow as tp
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, url_for, redirect, render_template, request


templates="templates/"
static="static/"

app = Flask(__name__, template_folder = templates , static_folder=static)

@app.route("/upload", method=['GET', 'POST'])

def upload():
    return render_template('upload.html')

app.run()