from pydantic import BaseModel, Field
from typing import Literal, Optional, List

class MoodAttributes(BaseModel):
    mood: Literal["joyful", "sad", "angry", "neutral", "confused", "excited", "anxious", "surprised", "disgusted", "fearful"] = Field(description="The primary emotion detected in the user's message.")
    intensity: Literal["low", "medium", "high"] = Field(description="The intensity of the detected mood.")
    reason: Optional[str] = Field(None, description="The likely reason for the user's mood, if discernible.")

class IntentAttributes(BaseModel):
    intent: Literal["question", "statement", "request", "command", "complaint", "greeting", "farewell", "expression"] = Field(description="The primary communicative intention of the user's message.")
    target: Optional[str] = Field(None, description="The specific subject or object of the user's intent, if applicable (e.g., 'weather' for a question, 'assistance' for a request).")
    details: Optional[str] = Field(None, description="Any additional specific details about the user's intent.")

class UserProfile(BaseModel):
    name: Optional[str] = Field(None, description="The user's name.")
    # interests: List[str] = Field(default_factory=list, description="A list of the user's stated interests or hobbies.")
    # preferences: List[str] = Field(default_factory=list, description="A list of other personal preferences of the user.")
    summary: Optional[str] = Field(None, description="A brief summary of the user's overall personal profile and psychological portrait.")

# New Pydantic model for controversial topic detection
class ControversialTopicAttributes(BaseModel):
    is_controversial: bool = Field(description="True if the message discusses a controversial topic (politics, religion, sexual, violence, hate speech), False otherwise.")
    category: Optional[Literal["politics", "religion", "sexual", "violence", "hate_speech", "none"]] = Field(
        None, description="The category of the controversial topic if detected, or 'none' if not controversial."
    )
    reason: Optional[str] = Field(None, description="A brief explanation if the topic is controversial.")

