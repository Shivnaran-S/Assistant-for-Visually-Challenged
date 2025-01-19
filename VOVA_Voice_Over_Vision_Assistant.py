import re #regular expression
from groq import Groq #AI company made a Language Processing Unit - 25X faster than GPU based systems like chatgpt
                      #Allows us to run large open source llms' - Llama 3
import google.generativeai as genai
from faster_whisper import WhisperModel
import os

from gtts import gTTS #google Text-To-Speech API. gtts - name of Python library and gTTS - class name
from IPython.display import Audio #Interactive Python display - module provides utilities for displaying rich media outputs in an IPython environment,
                                  #such as Jupyter notebooks or IPython shells.
                                  #Audio - is a class that is used to embed and play audio files directly within the notebook or IPython shell.

from IPython.display import display, Javascript
from IPython.display import Image as ipython_display_image #To avoid ambiguity
from google.colab.output import eval_js
from base64 import b64decode

from mutagen.mp3 import MP3 #To find the duration of the audio file
import time #time.sleep() to pause execution for the duration of the audio

from PIL import Image #From Python Imaging Library

wake_word = "hey"

# Initialize models and settings
groq_client = Groq(api_key="")  # Replace with your Groq API key
genai.configure(api_key='')  # Replace with your Generative AI API key
#openai_client = OpenAI(api_key='')  # Replace with your OpenAI API key
whisper_model = WhisperModel("base", device='cpu', compute_type='int8')

# System message for AI
sys_msg=(
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]


generation_config = {
    'temperature': 0.7, # LLM - a parameter that controls the randomness of the model's output and how creative it is
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)


def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n  IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are a vision analysis AI designed to derive semantic meaning from images and provide contextual information.'
        'Your purpose is to analyze the image based on the user\'s prompt and extract all relevant details.'
        'Your output will serve as input for another AI that will respond to the user. Do not respond directly to the user.'
        f'Instead, focus on generating objective and detailed data about the image that aligns with the user\'s prompt. \nUSER PROMPT: {prompt}'

    )
    response = model.generate_content([prompt, img])
    return response.text

def function_call(prompt):
    # System instruction
    # Function calling model - The model interprets the input, maps it to a predefined function, and returns the result.
    sys_msg = (
        'You are an AI function calling model. You will determine whether capturing the webcam or calling no functions is best for'
        'a voice assistant to respond to the user\'s prompt. If the user asks to capture the webcam then reply with the corresponding action.'
        'The webcam can be assumed to be a normal laptop webcam facing the user.'
        'You will respond with only one action from this list: ["capture webcam", "None"].'
        'Respond with any one of the function call name exactly as I listed based on the user\'s prompt.'
    )

    # Function Conversation
    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]
    # LLaMA - Large Language Model Meta AI. 3-rd generation 70 billion parameters and a max of 8192 tokens in the tokenized prompt
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')

    # response.choices : This is an array containing one or more possible responses generated by the model.
    '''
    {
      "id": "cmpl-xyz123",
      "object": "chat.completion",
      "created": 1678901234,
      "model": "llama3-70b-8192",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant", #AI assistant is responding/ if "role": "user" then user's query/ if "role": "system" then instruction guiding the assistant
            "content": "Take a screenshot."
          },
          "finish_reason": "stop"
        }
      ],
      "usage": {
        "prompt_tokens": 56,
        "completion_tokens": 8,
        "total_tokens": 64
      }
    }
    '''
    response = chat_completion.choices[0].message
    return response.content

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'
    # re - regular expression

    # '\' - treated as a escape/special character like '\t', '\n' ; '\\' - treated as a single backslash or a literal character not an escape character
    # usually '\' are treated as escape/special characters to escape them from being escape characters we use raw strings or double backslashes
    # r - makes it a raw string which treats '\' as literal character and do not interpret them as escape character(which escapes n from character to newline)

    # f - formatted string literal
    # name = "Bob"
    # print(f"Hello, {name}!")  # Output: Hello, Bob!

    # \b - word boundary, ensures wake_word is matched as a complete word not part of any other word
    # Ensures that "hi" is a whole word and not part of another word (e.g., won't match "higher").

    # special regex characters :- [], ., *, ()
    # re.escape(wake_word) - escapes any special regex characters in wake_word, and treat them as plain text in the regex

    # [] - character class. matches multiple characters but one character at a time from the character class
    # \s - any whitespace characters - spaces, tabs, newlines, carriage returns

    # * - zero or more of the preceding element
    # () - Groups sequences or patterns
    # . - matches any single character except a newline by default

    #Ultimate Aim :
    #Find the wake word as a whole word at the start of a segment in the input text.
    #Handle optional punctuation or whitespace immediately after the wake word.
    #Capture and extract the text that follows the wake word.

    match = re.search(pattern, transcribed_text, re.IGNORECASE) #makes the search case insensitive

    if match:
        prompt = match.group(1).strip()
        # If a match is found, this line extracts the first capturing group (group 1) from the match using match.group(1).
        # The capturing group is defined by parentheses () in the regular expression pattern.
        # The .strip() method removes any leading and trailing whitespace from the extracted string.
        return prompt
    else:
        return None

def process_audio_file(audio_file_path):
    prompt_text = wav_to_text(audio_file_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt:
        action = function_call(clean_prompt)
        #print('Function call response:', response)

        visual_context = None  # Initialize visual context variable
        if 'capture webcam' in action:
            capture_webcam()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='photo.jpg')

        response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
        return response
    else:
        print('No valid clean prompt found.')
        return None

def text_to_speech(text):
    tts = gTTS(text=text, lang='en') # Creating an object tts for the class gTTS by passing the values for class members
    tts.save("tts.mp3")

    audio_info = MP3("tts.mp3")
    duration = audio_info.info.length  # Duration in seconds

    display(Audio("tts.mp3", autoplay=True)) # Embedded audio player widget in the notebook interface

    # Wait until the audio completes
    time.sleep(duration)

def record_audio(filename='recorded_audio.wav'):
    # Javascript class from IPython

    # recordAudio() is Javascript's asynchronous function, helps to use await keyword which pauses the execution of the function until a promise resolves
    # Javascript is single threaded, it can only execute one operation at a time. To handle tasks like waiting for user input, fetching data, or recording audio
    # (which take time), JavaScript uses asynchronous operations so that it doesn’t block other tasks.

    # div - division - to divide or section off areas of a webpage so that they can be managed or styled independently.
    # Document is an object that represents the entire HTML page. lets you interact with and manupulate page's elements

    # Creates a new <p> (paragraph) element in the DOM and assigns it to the variable instructions, allowing you to modify or add it to the page.

    # Creates a <textarea> element for user interaction. rows and cols specify the size of the input box. placeholder provides a hint text in the box.

    js = Javascript('''
    async function recordAudio() {
        const div = document.createElement('div');
        const instructions = document.createElement('p');
        instructions.textContent = "Press Enter to start recording. Type '.' to stop.";

        const textarea = document.createElement('textarea');
        textarea.rows = 2;
        textarea.cols = 30;
        textarea.placeholder = "Press Enter to start...";

        div.appendChild(instructions);
        div.appendChild(textarea);
        document.body.appendChild(div);

        // Focuses on the textarea by automatically placing the cursor in the input box
        textarea.focus();

        // Execution of the code waits until the user presses the Enter key to start recording.
        // A Promise is an object representing an asynchronous task that will either complete successfully or fail.
        // await - asynchronous wait - Pauses the execution of the async function until the Promise is resolved.

        // Attaches an event listener to the textarea element.
        // Listens for a keydown event (when a key is pressed) and executes the callback function whenever the event occurs.
        // e: Represents the event object, which provides details about the keypress.

        await new Promise(resolve => {
            textarea.addEventListener('keydown', (e) => {
                if (e.key === "Enter") {
                    e.preventDefault(); // Prevent default behavior of the "Enter" key (e.g., inserting a new line in the textarea).
                    resolve(); // Ends the promise or marks the Promise as completed (resolved), signaling the async function to continue execution.
                }
            });
        });

        instructions.textContent = "Recording... Type '.' to stop.";

        // navigator.mediaDevices-This is an object provided by the browser to allow access to media input devices like microphones, cameras and screen sharing
        // Web Real Time Communication (Web-RTC) API

        // .getUserMedia({ audio: true }) - asynchronous operation that requests permission to access the user\'s media devices
        // The argument { audio: true } specifies that the application only needs access to the user\'s microphone (audio input).

        // await - pauses the execution of the async function until the getUserMedia promise resolves.

        // stream - this object contains the live audio data from the user\'s microphone, which can be used by other parts of the application
        //          (e.g., recording or processing the audio).
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // MediaRecorder is a built-in JavaScript interface provided by the browser. It is used to record audio and/or video streams.
        // It takes a media stream (like the one from a microphone or camera) and encodes it into chunks of data.
        // Recorder holds the instance the object
        const recorder = new MediaRecorder(stream);
        const chunks = [];

        // ondataavailable allows you to access the recorded data chunks, The ondataavailable event fires periodically as the recorder gathers audio data.
        // the below is an event handler which triggers when the MediaRecorder has a new piece of recorded data "chunk" available
        recorder.ondataavailable = e => chunks.push(e.data);

        // starts the recording process
        // When this method is called:
        // The MediaRecorder begins capturing audio from the stream provided during its initialization.
        // As the recording progresses, the ondataavailable event fires whenever a chunk of audio data is ready, which is then added to the chunks array.
        recorder.start();

        // Wait for '.' key to stop recording
        await new Promise(resolve => {
            textarea.addEventListener('keydown', (e) => {
                if (e.key === ".") {
                    e.preventDefault(); // Prevent default behavior
                    recorder.stop();
                    resolve();
                }
            });
        });

        // This wraps the recording process in a Promise. It allows you to wait for the audio recording to finish and for the processed audio data to become available.
        // await - Ensures that the code waits until the Promise resolves before continuing execution.
        // resolve - Function provided by the Promise. When called, it resolves the Promise and passes the resolved value (the audio data in this case) to audioData.

        const audioData = await new Promise(resolve => {
            recorder.onstop = async () => {       // Sets up an event handler for when the recording stops (triggered by recorder.stop()).
                                                  // When the recording stops, this function is executed.
                const blob = new Blob(chunks);
                // Combines the recorded audio chunks (stored in the chunks array) into a single Blob object. Is a raw data object that can hold multimedia data (e.g., audio, video).
                // BLOb - Binary Large Object

                const reader = new FileReader(); // Creates a FileReader object to read the contents of the Blob.

                reader.onload = () => resolve(reader.result.split(',')[1]);
                // reader.onload - Event handler that triggers when the FileReader finishes reading the Blob.
                // reader.result - Contains the data from the Blob as a Base64-encoded string.
                // .split(',')[1] - Extracts only the Base64 part of the data URL (removing the data:audio/wav;base64, prefix).
                // resolve(...) - Passes the Base64-encoded audio data to the Promise, resolving it.

                reader.readAsDataURL(blob);
                // starts reading the Blob as a Base64-encoded string
                // this triggers the onload event when finished.
                // Base64 encoded string is a textual representation of binary data that uses only 64 ASCII characters

            };
        });

        stream.getTracks().forEach(track => track.stop());
        // stream.getTracks() retrieves all the media tracks associated with the MediaStream object (in this case, the audio track).
        // .forEach(track => track.stop()) iterates through each track and stops it using the track.stop() method.
        div.remove(); // div which contains the instructions and the text area is removed from the DOM (Document Object Model) using the .remove() method.
        return audioData; // Returns the Base64-encoded audio data to the calling code.
    }
    ''')

    display(js) # This line displays the JavaScript code (js) in the frontend of a Jupyter Notebook or Google Colab environment.
                # renders and executes this JavaScript code in the browser.
    data = eval_js('recordAudio()') # Executes the recordAudio() function defined in the JavaScript code and retrieves its output.

    # Decode the audio data and save it as a WAV file
    audio_bytes = b64decode(data)
    with open(filename, 'wb') as f:
        f.write(audio_bytes)

    print(f"Audio saved to {filename}")
    return filename

def capture_webcam(filename='photo.jpg', quality=1.0):
    text_to_speech("As per your request, now the image will be captured through the web camera. So be ready.")
    time.sleep(0.25)
    text_to_speech('3')
    time.sleep(0.25)
    text_to_speech('2')
    time.sleep(0.25)
    text_to_speech('1')

    #Embeds a block of JavaScript code as a Python string using triple quotes ('''...''').
    js = Javascript('''
      async function takePhoto(quality) { //asynchronous function. async - ensures the function can handle await statements for asynchronous operations.
        const video = document.createElement('video'); //Creates an HTML <video> element using document.createElement.
        video.style.display = 'block'; //Sets the CSS style of the video to be displayed as a block, ensuring it appears on its own line.
        const stream = await navigator.mediaDevices.getUserMedia({video: true}); //API to request access to the user's webcam alone not audio and all

        document.body.appendChild(video); //Appends the video element to the body of the HTML document, making it visible in the browser.
        video.srcObject = stream; //Sets the video source to the stream obtained from the webcam.
        await video.play(); //starts streaming the webcam feed.

        // Resizes the Google Colab notebook's output cell to fit the height of the content (the displayed video feed).
        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true); //calculates the required height for the content.

        // Wait for 2 seconds before capturing the image
        await new Promise(resolve => setTimeout(resolve, 2000));

        //Creates an HTML <canvas> element to capture a single frame from the video feed.
        //Sets the canvas dimensions to match the resolution of the video (video.videoWidth and video.videoHeight).
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        canvas.getContext('2d').drawImage(video, 0, 0); //Captures the current frame of the video and draws it onto the canvas using the 2D drawing context.
        stream.getVideoTracks()[0].stop(); //Stops the first video track in the media stream, turning off the webcam.

        video.remove();  //Removes the video element from the HTML document after capturing the image to clean up the UI.

        return canvas.toDataURL('image/jpeg', quality); //Converts the canvas content (the captured frame) into a Base64-encoded JPEG image string.
      }
    ''')

    display(js)
    data = eval_js('takePhoto({})'.format(quality))

    display(Audio("camera_shutter.mp3", autoplay=True))
    time.sleep(2)

    #Base64 data URI of an image or audio file, which typically looks like this:"data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAAAAAA..."
    #After split(','), the string becomes: ['data:image/jpeg;base64', '/9j/4AAQSkZJRgABAQEAAAAAA...']
    #Accessing [1] extracts the Base64-encoded data: '/9j/4AAQSkZJRgABAQEAAAAAA...'
    binary = b64decode(data.split(',')[1])

    with open(filename, 'wb') as f:
        f.write(binary)

    text_to_speech("Thank you! The image has been successfully saved.")
    print(filename)
    display(ipython_display_image(filename))
    return filename

def main():
    '''
    vova = ('Hello! I am VOVA Voice Over Vision Assistant designed to serve visually challenged people. I am pleased to help you.'
            'If you want to query me, start by entering the ENTER key which will be usually present at the bottom right of QWERTY keyboard.'
            'If you are not interested in the conversation anymore you can quit by entering the dot . which is to the bottom left of ENTER key.'
           )#Implicit String Concatenation
    '''
    while True:
      user = input()
      if user=='.':
        text_to_speech('Since you entered dot, hope you don\'t have any questions. It was a nice time with you. Thank you so much.')
        break
      elif user=="":
        #text_to_speech('Since you entered the enter key, feel free to ask any questions by entering the Enter key and stop recording by entering the dot key')

        user = record_audio()
        audio = Audio('/content/recorded_audio.wav')
        display(audio)

        response = process_audio_file('/content/recorded_audio.wav')
        if response:
          print(response)
          text_to_speech(response)

      text_to_speech('I would like to respond you with some more questions. If you are interested in asking some more questions, please enter the Enter key, else if you are done with it, please enter dot, and the Enter key')


if __name__ == "__main__":
  main()
#__name__ is a special variable in Python that indicates how a script is being run:
#If the script is run directly, __name__ is set to "__main__".
#If the script is imported as a module, __name__ is set to the module's name.

# gemini-ai for image description
# gen ai - google.generativeai for image
# groq for text prompt and reply
# Open-ai's whisper model for transcription
