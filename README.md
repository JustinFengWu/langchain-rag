Steps to run the bot:
IMPORTANT NOTE: An OpenAI API Key is required

step 1: set your OpenAI API key as an environemental variable using the following commands:
    set MY_VARIABLE=my_value    ---->   Windows
    export MY_VARIABLE=my_value ---->   Linux
    export MY_VARIABLE=my_value ---->   MacOs

Step 2: activate virtual environemnt
    In the terminal run the activate file in .venv folder
    e.g.,
    ./.venv/Scripts/activate

Step 3: Run the Chatbot:
    execute the run.py file. If requirements not met, i.e., unnamed Module/Failed Import error, go to step 4
    Upon starting execution, please wait for a moment for the webcam emotion detection to startup, before recording
    voice. To terminate program properly once finished, follow step 6.

Step 4:
    install all dependencies in cahtbot_requirement.txt using the following:
        pip install -r chatbot_requirement.txt
    This should be every essential library required
    If modules still missing, go to step 5

Step 5:
    Install every dependency that exists in the developement requirement. (Note: it is around 200 lines long)
        pip install -r requirements1.txt
    This installs every library and dependency that was in the developement environment. Keep in mind that some
    of these dependecy are not necesary for this project.

Step 6:
    To terminate the program, press 'esc' first in the terminal. The conversation history will be saved, with 
    a message that will be displayed.
    Then go to the window demonstrating your webcam output. Press q to terminate the program.
    