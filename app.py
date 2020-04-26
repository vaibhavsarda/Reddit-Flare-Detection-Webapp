from flask import Flask,render_template,url_for,request,jsonify
import praw
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
from sklearn.ensemble import RandomForestClassifier as RFC
import json

def preprocess_text(s):
	
	if(type(s)==float):
		return "NaN"
	
	stop_words=set(stopwords.words('english'))
	tokenizer=RegexpTokenizer(r'\w+')
	lem=WordNetLemmatizer()
  
	word_tokens=tokenizer.tokenize(s)
	pp_content=""

	for g in word_tokens:

		if(g not in stop_words):
			pp_content=pp_content+lem.lemmatize(g.lower())+" "
	
	return pp_content

def TfIdf_Vectorization(combined_data):
	
	vectorizer=pickle.load(open('vect_object.pickle','rb'))
	X=vectorizer.transform(combined_data)
	return X

app=Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	saved_model=pickle.load(open('rf_95_37.sav','rb'))

	if(request.method=="POST"):
		
		reddit = praw.Reddit(client_id='dxCAxrQivxyfDw',\
					client_secret='GjWF0Jl0OK6-A5ihiqSAOF0xrLQ',\
					user_agent='Reddit Flare Detection',\
					username='vaibhavsarda',\
					password='Ganesh@108')

		msg_url=request.form['message']
		submission=reddit.submission(url=msg_url)

		all_data=preprocess_text(submission.title)

		all_comments=""
		
		# tlc: top level comments
		for tlc in submission.comments:
			all_comments=all_comments+" "+tlc.body

		all_data=all_data+preprocess_text(all_comments)		 

		data_arr=[all_data]

		X = TfIdf_Vectorization(data_arr)

		pred_flare=saved_model.predict(X)

	return render_template('result.html',prediction=pred_flare)

@app.route('/automated_testing',methods=['POST'])
def index():
	
	saved_model=pickle.load(open('rf_95_37.sav','rb'))

	if(request.method=='POST'):

		reddit = praw.Reddit(client_id='dxCAxrQivxyfDw',\
					client_secret='GjWF0Jl0OK6-A5ihiqSAOF0xrLQ',\
					user_agent='Reddit Flare Detection',\
					username='vaibhavsarda',\
					password='Ganesh@108')

		file=request.files['upload_file']
		fname=file.filename

		query_response={}

		f=open(fname,'rb')
		
		for re_ul in f:

			reddit_url=re_ul.decode('utf-8').strip()

			submission=reddit.submission(url=reddit_url)

			all_data=preprocess_text(submission.title)

			all_comments=""
			
			# tlc: top level comments
			for tlc in submission.comments:
				all_comments=all_comments+" "+tlc.body

			all_data=all_data+preprocess_text(all_comments)		 

			data_arr=[all_data]

			X = TfIdf_Vectorization(data_arr)

			pred_flare=saved_model.predict(X)

			query_response[reddit_url]=pred_flare[0]
			
		return jsonify(query_response)

if __name__=="__main__":
	app.run(debug=True)