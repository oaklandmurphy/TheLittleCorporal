import ollama
import json
import threading
from typing import Optional, Callable

class General:
	def __init__(self, faction: str, identity_prompt, unit_list=None, model: str = "llama3.2:3b", ollama_host: str = None):
		"""
		llm_command: List[str] - The command to run the local LLM (e.g., ["ollama", "run", "mymodel"])
		ollama_host: str - The Ollama API host URL (e.g., "http://localhost:11434")
		"""
		self.llm_command = ["ollama", "run", model]
		self.model = model
		self.ollama_host = ollama_host
		if ollama_host:
			self.client = ollama.Client(host=ollama_host)
		else:
			self.client = ollama
		self.name = identity_prompt.get("name", "General")
		self.description = identity_prompt.get("description", "")
		self.faction = faction
		self.unit_list = unit_list
		self.unit_summary = self.update_unit_summary()

	def get_instructions(self, player_instructions="", map_summary="", callback: Optional[Callable[[str], None]] = None):
		"""
		Passes player instructions and map summary to the LLM and returns the LLM's response as a general.
		If callback is provided, the query runs in a background thread and callback is called with the result.
		Otherwise, blocks until result is available.
		"""
		system_prompt, prompt = self._build_prompt(player_instructions, map_summary)
		
		if callback:
			# Run asynchronously in a background thread
			def run_query():
				result = self._query_general(system_prompt, prompt)
				callback(result)
			
			thread = threading.Thread(target=run_query, daemon=True)
			thread.start()
			return None  # callback will receive result
		else:
			# Synchronous call (for backward compatibility)
			return self._query_general(system_prompt, prompt)

	def _build_prompt(self, player_instructions, map_summary):
		"""
		Builds a prompt for the LLM to act as a battlefield general.
		"""
		if player_instructions.strip() == "":
			player_instructions = "You have received no orders, act according to your best judgement."

		system_prompt = (
			f"You are {self.name}, do not break character under any circumstances.\n"
			f"{self.description}\n"
			"Given the following battlefield summary and orders from the user, respond with clear, concise orders for your troops.\n"
			f"Battlefield Summary:\n{map_summary}\n"
			f"Your response must be in the form of a list of one line, direct orders to each of the following {len(self.unit_list)} units ({', '.join([unit.name for unit in self.unit_list])}) and nothing else.\n"
			f"You should reference exactly one location on the battlefield in each of your {len(self.unit_list)} orders."
		)

		prompt = f"Your orders are: {player_instructions}\n"

		return system_prompt, prompt

	def update_unit_summary(self):
		"""
		Generates a summary of the general's units.
		"""
		if not self.unit_list:
			return "No units assigned."
		summaries = []
		for unit in self.unit_list:
			summaries.append(f"{unit.status_general()}\n")
		return "\n".join(summaries)

	def _query_general(self, system_prompt, prompt, num_thread=4, num_ctx=4096):
		"""
		Sends the prompt to the local LLM using the ollama Python library and returns its response.
		Limits CPU and RAM usage by setting num_thread and num_ctx.
		"""
		try:
			# Compose system message with personality/system_prompt info
			system_message = {
				"role": "system",
				"content": system_prompt
			}
			user_message = {"role": "user", "content": prompt}
			response = self.client.chat(
				model=self.model,
				messages=[system_message, user_message],
				options={"num_thread": num_thread, "num_ctx": num_ctx}
			)
			return response["message"]["content"].strip()
		except Exception as e:
			return f"[LLM Error] {e}"