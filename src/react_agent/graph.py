"""Chatbot de recommandation de voyages."""

import json
import os
from typing import Optional, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
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
# Voyages
#=========================================================================================
VOYAGES = [
    {
        "nom": "Randonnée camping en Lozère",
        "labels": ["sport", "montagne", "campagne"],
        "accessibleHandicap": "non",
    },
    {
        "nom": "5 étoiles à Chamonix option fondue",
        "labels": ["montagne", "detente"],
        "accessibleHandicap": "oui",
    },
    {
        "nom": "5 étoiles à Chamonix option ski",
        "labels": ["montagne", "sport"],
        "accessibleHandicap": "non",
    },
    {
        "nom": "Palavas de paillotes en paillotes",
        "labels": ["plage", "ville", "detente", "paillote"],
        "accessibleHandicap": "oui",
    },
    {
        "nom": "5 étoiles en rase campagne",
        "labels": ["campagne", "detente"],
        "accessibleHandicap": "oui",
    },
]

#=========================================================================================
# Tool
#=========================================================================================
def find_matching_voyages(voyages: list, criteres: dict) -> list:
    matching = []
    for voyage in voyages:
        labels = voyage["labels"]
        accessible = voyage["accessibleHandicap"] == "oui"
        match = True
        for critere, valeur in criteres.items():
            if critere == "acces_handicap":
                if valeur is True and not accessible:
                    match = False
                    break
            else:
                if valeur is True and critere not in labels:
                    match = False
                    break
                if valeur is False and critere in labels:
                    match = False
                    break
        if match:
            matching.append(voyage["nom"])
    return matching


@tool
def rechercher_voyages(criteres: dict) -> list:
    """Recherche les voyages correspondant aux critères de l'utilisateur.
    Retourne la liste des noms de voyages correspondants.
    Doit être appelé après chaque mise à jour des critères."""
    return find_matching_voyages(VOYAGES, criteres)


#=========================================================================================
# Schémas Pydantic
#=========================================================================================
class Criteres(BaseModel):
    plage:          Optional[bool] = Field(default=None, description="L'utilisateur aime la plage")
    campagne:       Optional[bool] = Field(default=None, description="L'utilisateur aime la campagne")
    paillote:       Optional[bool] = Field(default=None, description="L'utilisateur veut un accès à des paillottes")
    montagne:       Optional[bool] = Field(default=None, description="L'utilisateur aime la montagne")
    ville:          Optional[bool] = Field(default=None, description="L'utilisateur aime la ville")
    sport:          Optional[bool] = Field(default=None, description="L'utilisateur veut faire du sport")
    detente:        Optional[bool] = Field(default=None, description="L'utilisateur veut se détendre")
    acces_handicap: Optional[bool] = Field(default=None, description="L'utilisateur a besoin d'un accès handicap")


class CriteresResponse(BaseModel):
    criteres:         Criteres      = Field(description="Critères mis à jour selon le message de l'utilisateur")
    incomprehensible: bool          = Field(default=False, description="True si le message est incompréhensible")
    ai_message:       Optional[str] = Field(default=None, description="Message poli à retourner si le message est incompréhensible")


#=========================================================================================
# State
#=========================================================================================
class InputState(TypedDict):
    user_message: str


class State(TypedDict):
    user_message:     str
    ai_message:       str
    criteres:         dict
    incomprehensible: bool


#=========================================================================================
# Prompts
#=========================================================================================
PROMPT_CRITERES = """Tu es un assistant qui extrait les critères de voyage d'un utilisateur.

Critères actuels :
{criteres}

Si le message est incompréhensible :
- Mets incomprehensible à True
- Rédige un message poli dans ai_message pour demander à l'utilisateur de reformuler
- Ne modifie pas les critères

Sinon, mets à jour uniquement les critères explicitement mentionnés :
- Envie positive                                        → True
- Annulation (ex: "oublie le sport", "retire le sport") → None
- Rejet      (ex: "je ne veux pas de sport")            → False
- Non mentionné → conserve la valeur actuelle (None si encore inconnu)"""


PROMPT_REPONSE = """Tu es un agent de recommandation de voyages pour une agence de tourisme.

Critères actuels de l'utilisateur :
{criteres}

## Règles
1. Appelle TOUJOURS l'outil rechercher_voyages avec les critères actuels.
2. Aucun critère à True     → demande à l'utilisateur de préciser ses envies.
3. Des voyages correspondent → affiche la liste retournée par l'outil.
4. Aucun voyage ne correspond → indique-le clairement et propose de modifier les critères.

Rappelle toujours en début de réponse le résumé des critères à True.
Réponds toujours en français."""

#=========================================================================================
# Modèles
#=========================================================================================
_llm_criteres = init_chat_model(
    model="mistralai/mistral-small-2603",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
).with_structured_output(CriteresResponse)

_llm_response = init_chat_model(
    model="mistralai/mistral-small-2603",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
).bind_tools([rechercher_voyages])

#=========================================================================================
# Nœud 1 : mise à jour des critères
#=========================================================================================
async def update_criteres(state: State) -> dict:
    criteres_actuels = state.get("criteres", {})

    messages_criteres = [
        {"role": "system", "content": PROMPT_CRITERES.format(
            criteres=json.dumps(criteres_actuels, ensure_ascii=False),
        )},
        {"role": "user", "content": state["user_message"]},
    ]
    print("---- messages envoyés au LLM critères ----")
    print(json.dumps(messages_criteres, ensure_ascii=False, indent=2))

    response: CriteresResponse = await _llm_criteres.ainvoke(messages_criteres)

    if response.incomprehensible:
        print("---- message incompréhensible ----")
        return {
            "criteres":         criteres_actuels,
            "incomprehensible": True,
            "ai_message":       response.ai_message,
        }

    # Merge : on écrase uniquement les champs explicitement fournis par le LLM
    # (exclude_unset=True distingue "non mentionné" de "annulé → None")
    nouveaux_criteres = criteres_actuels.copy()
    for k, v in response.criteres.model_dump(exclude_unset=True).items():
        nouveaux_criteres[k] = v

    print("---- criteres mis à jour ----")
    print(json.dumps(nouveaux_criteres, ensure_ascii=False))

    return {
        "criteres":         nouveaux_criteres,
        "incomprehensible": False,
    }

#=========================================================================================
# Nœud 2 : génération de la réponse avec tool
#=========================================================================================
async def generate_response(state: State) -> dict:
    criteres = state.get("criteres", {})

    messages = [
        {"role": "system", "content": PROMPT_REPONSE.format(
            criteres=json.dumps(criteres, ensure_ascii=False),
        )},
        {"role": "user", "content": state["user_message"]},
    ]

    # Boucle ReAct : le LLM peut appeler le tool plusieurs fois
    while True:
        response = await _llm_response.ainvoke(messages)
        messages.append(response)

        if not response.tool_calls:
            break

        for tool_call in response.tool_calls:
            tool_result = rechercher_voyages.invoke(tool_call["args"])
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

    print("---- response.message -----")
    print(response.content)

    return {"ai_message": response.content}

#=========================================================================================
# Routing
#=========================================================================================
def route_after_criteres(state: State) -> str:
    if state.get("incomprehensible"):
        return END
    return "generate_response"

#=========================================================================================
# Graph
#=========================================================================================
builder = StateGraph(State, input_schema=InputState)
builder.add_node("update_criteres", update_criteres)
builder.add_node("generate_response", generate_response)
builder.add_edge("__start__", "update_criteres")
builder.add_conditional_edges("update_criteres", route_after_criteres)
builder.add_edge("generate_response", END)

graph = builder.compile(name="ReAct Agent")