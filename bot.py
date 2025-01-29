import requests
import user_agent
import telebot

tok = "7755708665:AAEOgUu_rYrPnGFE7_BJWmr8hw9_xrZ-5e0"
bot = telebot.TeleBot(tok)
us = user_agent.generate_user_agent()

def getTok():
    headers = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "accept-language": "ar-EG,ar;q=0.9,en-GB;q=0.8,en;q=0.7,ar-AE;q=0.6,en-US;q=0.5",
        "cache-control": "max-age=0",
        "priority": "u=0, i",
        "referer": "https://www.google.com/",
        "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent":us,
    }

    response = requests.get("https://www.blackbox.ai/", headers=headers,timeout=20)

    try:
        chat = response.text.split("chat-")[1][:7]
    except:
        print("unexcpected error bruh the web has been broked fuk try again!")
        return "FVByyio"
    else:
        return chat

def sendMess(mess,code):
    headers = {
        "accept": "*/*",
        "accept-language": "ar-EG,ar;q=0.9,en-GB;q=0.8,en;q=0.7,ar-AE;q=0.6,en-US;q=0.5",
        "content-type": "application/json",
        "origin": "https://www.blackbox.ai",
        "priority": "u=1, i",
        "referer": f"https://www.blackbox.ai/chat/{code}",
        "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": us,
    }

    json_data = {
        "messages": [
            {
                "id": code,
                "content": mess,
                "role": "user",
            },
        ],
        "agentMode": {},
        "id": code,
        "previewToken": None,
        "userId": None,
        "codeModelMode": True,
        "trendingAgentMode": {},
        "isMicMode": False,
        "userSystemPrompt": None,
        "maxTokens": 1024,
        "playgroundTopP": None,
        "playgroundTemperature": None,
        "isChromeExt": False,
        "githubToken": "",
        "clickedAnswer2": False,
        "clickedAnswer3": False,
        "clickedForceWebSearch": False,
        "visitFromDelta": False,
        "isMemoryEnabled": False,
        "mobileClient": False,
        "userSelectedModel": None,
        "validated": "00f37b34-a166-4efb-bce5-1312d87f2f94",
        "imageGenerationMode": False,
        "webSearchModePrompt": False,
        "deepSearchMode": False,
        "domains": None,
        "vscodeClient": False,
        "codeInterpreterMode": False,
    }

    response = requests.post(
        "https://www.blackbox.ai/api/chat", headers=headers, json=json_data,timeout=20u
    )

    return response.text.split("$~~~$")[2]

cod = getTok()

@bot.message_handler(commands=["start"])
def Welcome(mess):
	bot.reply_to(mess,"هلا ، انا بوت فايدتي ذكاء اصطناعي ، اكتب سؤالك قبله /chat وبرد عليك على طول!\nانا طورني الهقر ايرور\n@Error_3mk")
	
@bot.message_handler(commands=["chat"])
def Ai(mess):
	try:
		test = mess.text.split()[1]
		message = str(mess.text.split()[1:])
	except:
		bot.reply_to(mess,"يعم استخدم الامر زي الناس مثلا اكتب\n'/chat كيفك يقلبي'")
	else:
		messs ="".join(word for word in message)
		repl= sendMess(messs,cod)
		bot.reply_to(mess,str(repl))
bot.polling(timeout=20)