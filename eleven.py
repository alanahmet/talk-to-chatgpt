from elevenlabslib import *
    
def vocalize_text(text, voices_by_name):    
    user = ElevenLabsUser("your api key")
    voice = user.get_voices_by_name(voices_by_name)[0]  # This is a list because multiple voices can have the same name

    voice.generate_and_play_audio(text, playInBackground=False)
