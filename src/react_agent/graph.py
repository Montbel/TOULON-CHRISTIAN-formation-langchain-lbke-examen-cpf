"""Chatbot de recommandation de voyages."""

import json
import os
import sys
from typing import Optional, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

#=========================================================================================
# Init LangSmith
#=========================================================================================
print("Enabling LangSmith...")

api_key = os.environ.get("LANGSMITH_API_KEY")
if api_key:
    os.environ["LANGSMITH_ENDPOINT"] = "https://eu.api.smith.langchain.com"
    os.environ["LANGSMITH_TRACING"]  = "true"
    os.environ["LANGSMITH_PROJECT"]  = "TOULON-CHRISTIAN-formation-langchain-lbke-examen-cpf"
    print("LangSmith activated")
else:
    print("LangSmith is not activated (no API key)")

#=========================================================================================
# Données : voyages disponibles
#=========================================================================================
VOYAGES = [
    {
        "nom": "Randonnée camping en Lozère",
        "labels": ["sport", "montagne", "campagne"],
        "accessibleHandicap": "non",
    },
    {
        "nom": "5 étoiles à Chamonix option fondue",
        "labels": ["montagne", "détente"],
        "accessibleHandicap": "oui",
    },
    {
        "nom": "5 étoiles à Chamonix option ski",
        "labels": ["montagne", "sport"],
        "accessibleHandicap": "non",
    },
    {
        "nom": "Palavas de paillotes en paillotes",
        "labels": ["plage", "ville", "détente", "paillote"],
        "accessibleHandicap": "oui",
    },
    {
        "nom": "5 étoiles en rase campagne",
        "labels": ["campagne", "détente"],
        "accessibleHandicap": "oui",
    },
]

#=========================================================================================
# Schémas Pydantic pour la sortie structurée
#=========================================================================================
class Criteres(BaseModel):
    plage:          Optional[bool] = Field(default=None, description="L'utilisateur aime la plage")
    montagne:       Optional[bool] = Field(default=None, description="L'utilisateur aime la montagne")
    ville:          Optional[bool] = Field(default=None, description="L'utilisateur aime la ville")
    sport:          Optional[bool] = Field(default=None, description="L'utilisateur veut faire du sport")
    detente:        Optional[bool] = Field(default=None, description="L'utilisateur veut se détendre")
    acces_handicap: Optional[bool] = Field(default=None, description="L'utilisateur a besoin d'un accès handicap")


class AgentResponse(BaseModel):
    criteres: Criteres = Field(description="Critères mis à jour selon le message de l'utilisateur")
    message:  str      = Field(description="Réponse en langage naturel à envoyer à l'utilisateur")

#=========================================================================================
# State
#=========================================================================================
class InputState(TypedDict):
    user_message: str


class State(TypedDict):
    user_message: str
    ai_message:   str
    criteres:     dict  # sérialisation de Criteres

#=========================================================================================
# Prompt système
#=========================================================================================
SYSTEM_PROMPT = """Tu es un agent de recommandation de voyages pour une agence de tourisme.

Voyages disponibles :
{voyages}

Critères actuels de l'utilisateur :
{criteres}

Ton rôle :
1. Analyse le message de l'utilisateur et mets à jour les critères :
   - Goût positif pour un critère  → True
   - Goût négatif ou indifférence  → False
   - Critère non mentionné         → conserve la valeur actuelle (None si encore inconnu)
2. Réponds en un seul message :
   - Aucun critère à True : demande à l'utilisateur de préciser ses envies.
   - Au moins un critère à True : propose les voyages correspondants et invite à préciser.
3. Si le message est incompréhensible, signale-le poliment et continue le scénario.

Réponds toujours en français."""

#=========================================================================================
# Nœud principal
#=========================================================================================
async def call_model(state: State) -> dict:
    model = init_chat_model(
        model="mistralai/codestral-2508",
        model_provider="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    ).with_structured_output(AgentResponse)

    system_message = SYSTEM_PROMPT.format(
        voyages=json.dumps(VOYAGES, ensure_ascii=False, indent=2),
        criteres=json.dumps(state.get("criteres", {}), ensure_ascii=False),
    )

    response: AgentResponse = await model.ainvoke([
        {"role": "system", "content": system_message},
        {"role": "user",   "content": state["user_message"]},
    ])

    return {
        "ai_message": response.message,
        "criteres":   response.criteres.model_dump(),
    }

#=========================================================================================
# Graph
#=========================================================================================
builder = StateGraph(State, input_schema=InputState)
builder.add_node("call_model", call_model)
builder.add_edge("__start__", "call_model")
builder.add_edge("call_model", END)

graph = builder.compile(name="ReAct Agent")
