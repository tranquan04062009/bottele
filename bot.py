import os
import random
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup,  ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
from collections import Counter, deque
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from io import BytesIO
import json
import time
import sqlite3  # to persistent feedback history

# Lấy token từ biến môi trường
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("Vui lòng đặt biến môi trường TELEGRAM_TOKEN chứa token bot!")
BOT_NAME = "tài xỉu BOT"
DATABASE_NAME = "txbot_feedback.db"  # Define the database filename for history record
DATA_PERSISTENT_PATH=  "bot_state.json" #data is stored in case there are interruptions.


# Define model as a state
history_data = deque(maxlen=400) #used for state parameters of training.
train_data = [] #array of data set used for ML
train_labels = []
le = LabelEncoder() #encoding the parameters
scaler = StandardScaler()
# -- reinforcement parameters 
feedback_weights = {'correct': 1.0, 'incorrect': -0.5} # weight on each approach
strategy_weights = {'deterministic':0.8, 'cluster':0.5 ,  'machine_learning': 1.2,  'probability':0.4,  'streak':0.3 ,'statistical':0.2 }# used to evaluate weights
last_prediction = {'result' : None, 'strategy': None , 'model' : None  } # Used to hold last result
user_feedback_history=deque(maxlen = 1000 )# keep history in memory (for debugging, can save into file)



# --- Define Models --
model_logistic = LogisticRegression(random_state=42, solver='liblinear')
model_svm = SVC(kernel='linear', probability=True, random_state=42) 
model_sgd = SGDClassifier(loss='log_loss', random_state=42)
model_rf = RandomForestClassifier(random_state=42)
model_nb = GaussianNB()

model_calibrated_svm = CalibratedClassifierCV(model_svm, method='isotonic', cv=5)

model_kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
models_to_calibrate=[model_logistic,model_sgd, model_rf]
calibrated_models={}

# --------DATABASE Functions-----------
def create_feedback_table():
    """Creates the user feedback database"""
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_type TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
           
        )
    """)
    conn.commit()
    conn.close()
def save_user_feedback(feedback):
    """Saves user's Feedback """

    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO user_feedback (feedback_type) VALUES (?)", (feedback,))
    conn.commit()
    conn.close()



def load_data_state():
    """load bot status or data. from json (load on init ) if avalable  , defaults otherwise """

    global  strategy_weights, last_prediction, user_feedback_history , history_data

    if os.path.exists(DATA_PERSISTENT_PATH):

      with open(DATA_PERSISTENT_PATH, 'r') as f:
         loaded_data =json.load(f)
         #if some keys doesnt' exist, assume empty/ defaults. otherwise take what is avalable to it's variable state
         strategy_weights=loaded_data.get("strategy_weights",strategy_weights )  #take state/defaults  from existing , defaults in init, 
         last_prediction = loaded_data.get("last_prediction" , last_prediction )
         user_feedback_history = deque( loaded_data.get("user_feedback_history" , []),  maxlen = 1000  )
         history_data  =  deque ( loaded_data.get("history_data" , [])  ,  maxlen=400)   #defaults , otherwise take values previously stored


         print("loaded persistent data from previous session")


def save_data_state():
    """ saves to disk state for next boot of program."""

    global strategy_weights, last_prediction, user_feedback_history, history_data

    data= { # prepare an array of values before serializing them using json

       "strategy_weights": strategy_weights,
        "last_prediction":last_prediction,
        "user_feedback_history": list(user_feedback_history),
        "history_data": list(history_data) #serialize the current state in dict to save into a file, note that json dont handle complex datatypes so using casting as type (list(), tuple() if needed )

     }
    try :
        with open(DATA_PERSISTENT_PATH,'w') as f :
            json.dump(data,f)  #json write object
            print("Persistent bot's data is saved.")
    except Exception as e:
        print(f"Could not persist information as an error occurred {e}")


def save_current_history_image():
   """ Helper to capture current state from previous sessions"""

   if not history_data: return # no data, ignore 
   #  only called via /logchart  so no issues of None
   chart_image= generate_history_chart(history_data) # no data validation
  
   ts = time.time()  # using current time so we save images in proper name convention.

   name= "chart_tx_current_state_" + str(ts)+ ".png" #define a good file naming convetions to not conflict of previously captured images
  

   with open (name,"wb" ) as file:  # we store using "wb"  mode
       file.write(chart_image.read())   #we have to read  and pass as byte , in disk
   
   print ("saving ", name ," in current dir") # for feedback for saving info
    

# ----Visualization
def generate_history_chart(history):

    if not history:
        return None
    
    labels, values = zip(*Counter(history).items())  # Extract values

    plt.figure(figsize=(8, 5)) 
    plt.bar(labels, values, color=['skyblue', 'salmon'])

    for i, v in enumerate(values):
        plt.text(labels[i], v + 0.1, str(v), ha='center', va='bottom')


    plt.xlabel('Kết quả (t: Tài, x: Xỉu)', fontsize=12)
    plt.ylabel('Tần suất', fontsize=12)
    plt.title('Phân bố Tần suất kết quả', fontsize=14)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0) # Go to beginning of image bytes, readable by photo param from message in telegram
    plt.close() #release all previous plot charts for next run

    return buffer

# ----Mathematical Functions
def calculate_probabilities(history):
   
    if not history:
        return {"t": 0.5 , "x" : 0.5} 
  
    counter = Counter(history)
    total = len(history)

    prob_tai = counter["t"] / total
    prob_xiu = counter["x"] / total

    return  {"t": prob_tai , "x" : prob_xiu}

def apply_probability_threshold(prob_dict, threshold_t =0.54,threshold_x=0.46):

    return "t" if prob_dict["t"] > threshold_t else "x" if prob_dict["x"] > threshold_x else None # threshold check for prediction logic


# ---- Statistical Prediction functions -----
def statistical_prediction(history,  bias = 0.5):
    
    if not history:
        return random.choice(["t", "x"])
    counter = Counter(history)
    total = len(history)
    if total == 0:
       return random.choice(["t", "x"])
    
    prob_tai = counter["t"] / total
    prob_xiu = counter["x"] / total

    return "t" if (random.random() < prob_tai * (1+ bias) /2)  else "x" if  random.random()  < (prob_xiu * (1 + bias ) /2 )  else random.choice(["t", "x"])



#----- PreProcessing data and Scaling -----
def prepare_data_for_models(history):
        
        if len(history) < 5:
           return  None, None # if params below min.
    
        # Using 5 recent historical parameters for model params

        encoded_history_5 = le.fit_transform(history[-5:])
        features=np.array([encoded_history_5])
    
        
        X= scaler.fit_transform(features) #scale before any model.

        # last result of dataset will determine what next is predicted
        labels = le.transform([history[-1]])
        y=np.array(labels)
     
        return X , y #return dataset pre-processed

# ---Machine Learning ----

def train_all_models():
       
        if len(train_data) < 10: #return if no dataset available or less then 10
              return

        X=[]
        Y=[]

        for history in train_data:
            features , label = prepare_data_for_models(history) #getting pre-processed parameters


            if features is not None and label is not None:
                X.append(features[0]) 
                Y.append(label[0])

    
        if( len(X) >1 and len(Y) > 1):

         
           # calibrating using the data above from past iterations
           for model in models_to_calibrate:
              try:
                X=np.array(X) # cast array if it changes
                model.fit(X,Y)  # retrain/calibrate models that could do it

                calibrated_models[model] = model  #keep it to array ( so each one can hold itself with own state)

              except ValueError: # some models ( linear regressors might have valueError on low counts) skip them otherwise 
                 pass



           # after calibrated training, if proper parameters, training rest
           X=np.array(X)  #ensure data is proper numpy dataset for models that expect such format as parameters
           Y=np.array(Y)

           model_svm.fit(X,Y) #fit / train  using prepared data above and models pre-existing structure
           model_calibrated_svm.fit(X, Y)  # same, train also the calibrate SVM
 

def ml_prediction(history):

    if len(train_data) < 10:
       return statistical_prediction(history)


    features, label =prepare_data_for_models(history)  #retrieve features as pre-processed method call


    if features is None : #early exists checks with None , preventing unnesesary calculations down-stream
          return None



    # getting trained data
    model_svm_prob = model_calibrated_svm.predict_proba(features)
    svm_prediction_label = model_calibrated_svm.predict(features)
   

    log_prob, log_label =   _predict_probabilty(calibrated_models.get(model_logistic,model_logistic ),features)   #calling each one
    sgd_prob,  sgd_label=    _predict_probabilty(calibrated_models.get(model_sgd,model_sgd) ,features )  #retrive results, defaults if values are wrong or skip it due models not using that approach.
    rf_prob,  rf_label=      _predict_probabilty(calibrated_models.get(model_rf,model_rf),features )

    #prepare probablity for average/result outputs
    tai_probabilities_average=[]
    xiu_probabilities_average=[]

    if (not np.isnan(log_prob["t"])): tai_probabilities_average.append(log_prob["t"] ) #only append if the probabilities are numbers
    if (not np.isnan(sgd_prob["t"])) :tai_probabilities_average.append(sgd_prob["t"])
    if (not np.isnan(rf_prob["t"]))  : tai_probabilities_average.append(rf_prob["t"])

    if (not np.isnan(log_prob["x"]))  :xiu_probabilities_average.append(log_prob["x"]) #same checks before adding
    if (not np.isnan(sgd_prob["x"])) : xiu_probabilities_average.append(sgd_prob["x"])
    if (not np.isnan(rf_prob["x"])): xiu_probabilities_average.append(rf_prob["x"])


    average_prob_t= np.mean(tai_probabilities_average) if tai_probabilities_average else 0 #averaging results, defaults of '0' on issues
    average_prob_x= np.mean(xiu_probabilities_average) if xiu_probabilities_average else 0

    avg_probabilty=   {"t" :  average_prob_t,   "x":  average_prob_x   } #prepare value results,

    svm_label = le.inverse_transform(svm_prediction_label)[0]

    predicted_outcome= apply_probability_threshold(avg_probabilty, 0.52,0.48 ) #try the threshold
    if predicted_outcome : #If we do return result.
      return predicted_outcome
    else:

       return svm_label # if not threshold by probability from average. , defaults to use output from  svm ( Support vectors classifiers).



def _predict_probabilty(model, features):
     """ helper for models probabilties and labels, return both in dict format if method return predictions, or will return None dict if cannot."""
     
     if(  hasattr(model,'predict_proba')): #if probability for specific function avalable then retun, other skip
            try:
                probs = model.predict_proba(features)[0] #prediction if probability is supported method
                labels = le.inverse_transform(model.predict(features))   # using original transformation and model.

                prob_dictionary= dict(zip(le.classes_, probs))
                return prob_dictionary,labels[0]

            except ValueError :  #error protection during predictions return with None if there is an unhadled model behavior during predict or probabilistic outputs
                   return   {"t" :  float('NaN'), "x" : float('NaN')}, None # Return dict of not a number if models skip (eg loss fun not using probability as an output/result of calculations) 

     return   {"t" : float('NaN'), "x" : float('NaN')},None   # or return None by defaults for missing models prob outputs
def cluster_analysis(history):
        """ Apply KMeans clustering to determine data patterns"""

        if len(history) < 5 :
             return None
    
        encoded_history = le.fit_transform(history)
        reshaped_history = encoded_history.reshape(-1, 1)

        try:
           model_kmeans.fit(reshaped_history)   # trains kmeans data model, returns or error skips,
        except ValueError:  
            return None


        last_five = le.transform(history[-5:]) #encode using last results of historic
        last_five = last_five.reshape(1, -1) #reshape results
      
        if(model_kmeans.predict(last_five[0].reshape(-1,1))[0] == 0 ):  #predict value is group 0

            counter = Counter(history[-5:]) #checking frequencies of the given cluster group values for results t or x

            if counter["t"] > counter["x"]: #more of one , use the output based from which appear most frequent
              return 't'
            else: 
              return "x"
        elif(model_kmeans.predict(last_five[0].reshape(-1,1))[0] == 1):# same predict as in cluster number = 1, other wise will be handled down streams 
           if history[-1]=='t':  # use opposite (based by historical info).
               return 'x'
           else :
                return 't' #same principle if last value does appear ( it will swap or choose the oppisite label as final result based on clusters output ).

#---- Pattern Detections ----
def analyze_real_data(history):
 
    if len(history) < 3:  # if low result size ignore / or early exists by condition ( if small ).
        return None
    
    if all(item == history[0] for item in history):# same result through
        return history[0] #returns what the series appears as final result

    if all(history[i] != history[i + 1] for i in range(len(history) - 1)):#sequence t then x.. t then x and reverse with t-> x ,  or  x -> t based in this approach as oppposite.
        return "t" if history[-1] == "x" else "x"

    return None   # defaults to null if conditions above fail



#----Deterministic Rules---
def deterministic_algorithm(history):

    if len(history) < 4 : # if lower then condition
       return None

    if history[-1] == history[-2] == history[-3] and history[-1] == 't':
        return 'x' #if appear 't' then return the next as the inverse which 'x'.

    if history[-1] == history[-2] == history[-3] and history[-1] == 'x': #Same as if last results return the x or "opposite output for those sequences
         return 't' #same if appears x return "t" or  reverse.

    if history[-1] != history[-2] and history[-2] != history[-3] and history[-3] != history[-4]:  # check that diff pattern and swap value depending by pattern
         return "t" if history[-1] == "x" else "x"

    return None #return Nulls if fail checks


# ------ Reinforcement logic here -----------

def adjust_strategy_weights(feedback, strategy): #adjusting strategy
    global strategy_weights
    weight_change = feedback_weights.get(feedback, 0.0)  #gets result
    strategy_weights[strategy] += weight_change * strategy_weights[strategy] * 0.2  #update with percentage ( it also takes strategy output weight).

    strategy_weights[strategy] = min(max(strategy_weights[strategy], 0.1), 2.0)  #ensuring within range so values aren't crazy/
    #clamp weight to range to prevent extreme predictions on the strategy values
    return strategy_weights
#--------Core Prediction Strategy ------
def combined_prediction(history):
     global last_prediction
     strategy= None # strategy default


     # 1- Applying deterministic strategy with historical analysis.

     algo_prediction = deterministic_algorithm(history)

     if algo_prediction: #returns with a strategy name with each approach ( so the Reinforcement knows which method is applied).

           strategy = "deterministic"
           last_prediction.update({'strategy':strategy ,'result':algo_prediction  })

           return  algo_prediction

      # 2- Clusters
     cluster_prediction= cluster_analysis(history)

     if cluster_prediction :  # Same cluster method to get data output and returns  model and prediction strategy so RL engine/reinforcments
            strategy= "cluster"
            last_prediction.update({'strategy':strategy , 'result':cluster_prediction})
            return  cluster_prediction
    

   #3  machine Learning for any patterns not picked from above layers of deterministic approach

     ml_pred = ml_prediction(history)

     if ml_pred:

        strategy="machine_learning"
        last_prediction.update( { 'strategy': strategy,'result' : ml_pred}) # same principles (set what is currently using with prediction,  and the output method result  , for later adjustments in other logic functions
        return  ml_pred



    #4   Using  probability threshold ( from math/probalistic outputs methods using past sequences.  )

     probability_dict=calculate_probabilities(history) # if we do return based probability as preference

     probability_pred =  apply_probability_threshold(probability_dict) #try from methods for value using treshold if have results

     if(probability_pred) :
           strategy= "probability"

           last_prediction.update({'strategy':strategy,'result': probability_pred})

           return  probability_pred


     #5  Series/ Streak analysis with history patterns to see any known approach, use existing dataset  if methods dont have data.. or fail cases from others models

     streak_prediction = analyze_real_data(history)
     if streak_prediction:
           strategy = "streak"

           last_prediction.update({'strategy':strategy, 'result':streak_prediction })

           return  streak_prediction
 
      #5 If  all  methods above do fail, returns from the most basic and least performance approach .  as final prediction mechanism for output (using history patterns weighted bias approach as basic algorithm  output) .
     strategy="statistical"
     last_prediction.update({'strategy':strategy,'result': statistical_prediction(history, 0.25)})# default to random stat with historical bias.

     return statistical_prediction(history, 0.25)  # finally by return a statistical based bias random choice method


# -------- Bot Handlers---------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(f"Chào mừng bạn đến với {BOT_NAME} bot!\n"
        "Sử dụng /tx để dự đoán, /add để thêm kết quả.\n"
        "Nhập /help để xem hướng dẫn, /history to view history, or /chart to check result pattern, use /logchart for disk backup of patterns.\n",parse_mode = ParseMode.MARKDOWN)

# /tx command
async def tx(update: Update, context: ContextTypes.DEFAULT_TYPE):

        try:
            user_input = ' '.join(context.args) # get args

            if not user_input: #handle null args / skip early exists if user do not give arguments for the logic.
               await update.message.reply_text("Vui lòng nhập dãy lịch sử (t: Tài, x: Xỉu)!")
               return

            history = user_input.split()# transform using user input .
            if not all(item in ["t", "x"] for item in history): # check input sequence must include 'x' or 't' others are skipped and ask for more valid set

                 await update.message.reply_text("Dãy lịch sử chỉ được chứa 't' (Tài) và 'x' (Xỉu).")

                 return #Early skips.. return default errors
        
            history_data.extend(history)#using current sequence.



            #only save data set to train with more 5+ , prevent less-performant logic due params.
            if len(history) >= 5:
                  train_data.append(list(history_data))
                  train_labels.append(history[-1]) # saving by last result ( that models use ).

            train_all_models()

            # get result
            result = combined_prediction(list(history_data))
            last_prediction["model"] = BOT_NAME
            #add user controls/ feedback mechanism, after a single prediction has made
            keyboard = [
                [InlineKeyboardButton("Đúng", callback_data='correct')],
                [InlineKeyboardButton("Sai", callback_data='incorrect')]
             ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            #result and style for messages, after the full process have finish and feedback button are availible
            formatted_result=f"Kết quả dự đoán của {BOT_NAME} : *{'Tài' if result == 't' else 'Xỉu'}* "
            await update.message.reply_text( formatted_result,reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN) # using result by current set

        except Exception as e:  #general errors during command usage to capture exceptions from any place inside this method/functions
             await update.message.reply_text(f"Lỗi: {e}")

async def add(update: Update, context: ContextTypes.DEFAULT_TYPE):
     """ Handler of adding information in case some additional parameters were not used and users like to add results"""
     try:

            user_input = ' '.join(context.args) # get args

            if not user_input:
                   await update.message.reply_text("Vui lòng nhập kết quả thực tế (t: Tài, x: Xỉu)!")
                   return #Early exits to ignore and return errors, otherwise rest will throw more problems..
            new_data = user_input.split()
            if not all(item in ["t", "x"] for item in new_data): # check only parameters we support ( others return and report error and stop).
                await update.message.reply_text("Kết quả chỉ được chứa 't' (Tài) và 'x' (Xỉu).")
                return
           
            history_data.extend(new_data)# append from current to data set.


            #only train  models with some valid values on user dataset
            for i in range(len(new_data) - 5 + 1):
                   train_data.append(list(history_data))#adding user train-sets for machine learning, every single data after offset 5th value is added.
                   train_labels.append(new_data[i + 4])

            train_all_models() # if dataset ready do train data.
            await update.message.reply_text(f"Dữ liệu thực tế đã được cập nhật: {new_data}") # message update info after command processing complete, by each iteration

     except Exception as e:# catch general issues
       await update.message.reply_text(f"Lỗi : {e}")
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    feedback = query.data
    global  user_feedback_history # access values
    if last_prediction.get("strategy") is None or last_prediction.get('result') is None or  last_prediction.get('model')  is None :

       await query.edit_message_text("Không thể ghi nhận đánh giá, vì dự đoán không tồn tại, vui lòng thử lại sau.")
       return  #return as result do no exits yet.
    if feedback == 'correct':
       user_feedback_history.append({'result' : last_prediction['result'], 'strategy':last_prediction['strategy'],  'feedback':'correct', 'timestamp' : time.time() })

       save_user_feedback('correct')  #database
       await query.edit_message_text("Cảm ơn bạn! Đánh giá được ghi nhận.")
    elif feedback == 'incorrect':
        user_feedback_history.append({'result' : last_prediction['result'], 'strategy':last_prediction['strategy'],  'feedback':'incorrect'  ,'timestamp': time.time() })# append a history of feedback of each request for training adjustments for reinforcement learning.
        save_user_feedback('incorrect')
        await query.edit_message_text("Cảm ơn bạn! Tôi sẽ cố gắng cải thiện hơn nữa!")
   
    #Adjust weights strategy using a user feedback approach ( correct -> boost value or reduce for wrong , same goes for the opposite for other categories of feedback).

    adjust_strategy_weights(feedback, last_prediction["strategy"] )


async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):

    if not history_data: # check for  value sizes
        await update.message.reply_text("Chưa có dữ liệu lịch sử.") #early exits and skip
    else:

       await update.message.reply_text(f"Lịch sử: {' '.join(history_data)}")#otherwise get/show the last known record for dataset.


# helper for chart display ( by bot messages / text message handler).
async def chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """ /Chart output , bot responses """
    chart_image= generate_history_chart(history_data)  # generate image
    if chart_image is None:
       await update.message.reply_text("Không có dữ liệu lịch sử nào để hiển thị biểu đồ.")# return error for no dataset exists in method return a fail ( null values or small).

       return #early skip
    await update.message.reply_photo(photo=chart_image,caption='Historical Result Pattern') # showing to the user from the response in a single bot message (by image method support).
# helper for storing current history image in server's current folder by current date
async def logchart(update: Update, context: ContextTypes.DEFAULT_TYPE):

    """save current history to local filesystem to monitor """
    save_current_history_image()
    await update.message.reply_text("Đã lưu chart ở Server.")


# /help output. for display
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):

     await update.message.reply_text(
        f"Hướng dẫn sử dụng bot *{BOT_NAME}*:\n"
        "/tx [dãy lịch sử]: Dự đoán kết quả Tài/Xỉu.\n"
        "/add [kết quả]: Cập nhật kết quả thực tế.\n"
        "/history : Xem lịch sử gần đây.\n"
        "/chart : Check lịch sử chart ( đồ thị dữ liệu).\n"
         "/logchart : Lưu trữ lịch sử đồ thị vào Server .\n"

        "Ví dụ:\n"
        "- /tx t t x t x\n"
        "- /add t x x t t",parse_mode=ParseMode.MARKDOWN ) #using bot by methods support output style.

# ---------- bot setup main method to invoke --------
if __name__ == "__main__":
   create_feedback_table() # make database.

   load_data_state()#Load State from previous use

   app = ApplicationBuilder().token(TOKEN).build() # start the telegram framework

   #Commands bind to those methods handlers
   app.add_handler(CommandHandler("start", start))
   app.add_handler(CommandHandler("tx", tx))
   app.add_handler(CommandHandler("add", add))
   app.add_handler(CommandHandler("history", history))
   app.add_handler(CommandHandler("help", help_command))
   app.add_handler(CommandHandler("chart", chart))
   app.add_handler(CommandHandler("logchart",logchart ))


   app.add_handler(CallbackQueryHandler(button))  # add feedback handler using a button call in chat system.
   print("Bot starting , waiting...") # Bot startup messages before starting (logging the boot messages ).

   app.run_polling() #using polling
   save_data_state()# when close saves current states as local data-state for next boot cycle using the data state ( as file.json).

   print("Bot exiting with code = 0.")  #Exit Log message , once stop function in terminal ( or any shutdown)