{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "   WARNING: This is a development server. Do not use it in a production deployment.\n",
      "   Use a production WSGI server instead.\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:9800/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [05/Jul/2021 17:04:43] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, render_template, request\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import PassiveAggressiveClassifier\n",
    "import pickle\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "app = Flask(__name__)\n",
    "loaded_model = pickle.load(open('smodel.pkl', 'rb'))\n",
    "def requestResults(result):\n",
    "    if result == 0:\n",
    "        return \"Not-Sarcastic\"\n",
    "    else:\n",
    "        return \"Sarcastic\"\n",
    "def sardet(text):\n",
    "    tfvect = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000)\n",
    "    dataframe = pd.read_csv(\"train-balanced-sarcasm.csv\")\n",
    "    x = dataframe['comment'].astype('U').values\n",
    "    y = dataframe['label']\n",
    "    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)\n",
    "    tfid_x_train = tfvect.fit_transform(x_train)\n",
    "    tfid_x_test = tfvect.transform(x_test)    \n",
    "    input_data = [text]\n",
    "    vectorized_input_data =  tfvect.transform(input_data)\n",
    "    prediction = loaded_model.predict(vectorized_input_data)\n",
    "    return prediction\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('index.html')\n",
    "\n",
    "@app.route('/predict', methods=['POST'])\n",
    "def predict():\n",
    "    message = request.form['message']\n",
    "    pred = sardet(message)\n",
    "    print(pred)\n",
    "    result=requestResults(pred)\n",
    "    print(str(result))\n",
    "    return render_template('index.html', prediction_text=result)\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True,use_reloader=False, port=9800)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.7.0 64-bit ('base': conda)",
   "language": "python",
   "name": "python37064bitbasecondad777af226dbb42b881fd2d5fa43dee7a"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
