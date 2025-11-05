import ollama
import json

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
			if self.client is not None:
				print(f"Using remote Ollama host at {ollama_host}")
			else:
				print(f"Failed to connect to Ollama host at {ollama_host}, please check the host URL.")
		else:
			self.client = ollama
		self.name = identity_prompt.get("name", "General")
		self.description = identity_prompt.get("description", "")
		self.faction = faction
		self.unit_list = unit_list
		self.unit_summary = self.update_unit_summary()

	def get_instructions(self, player_instructions="", map_summary=""):
		"""
		Passes player instructions and map summary to the LLM and returns the LLM's response as a general.
		"""
		system_prompt, prompt = self._build_prompt(player_instructions, map_summary)
		generals_orders = self._query_general(system_prompt, prompt)

		return generals_orders

	def _build_prompt(self, player_instructions, map_summary):
		"""
		Builds a prompt for the LLM to act as a battlefield general.
		"""
		if player_instructions.strip() == "":
			player_instructions = "You have received no orders, act according to your best judgement."

		system_prompt = (
			f"You are {self.name}, do not break character under any circumstances.\n"
			f"{self.description}\n"
			"You command the following units:\n"
			f"{self.unit_summary}\n"
			"Given the following battlefield summary and orders from the user, respond with clear, concise orders for your troops.\n"
			f"Battlefield Summary:\n{map_summary}\n"
			"Your response must be in the form of a list of direct orders to each of your units and nothing else.\n"
			"you should try to reference a location on the map or a diffferent unit when giving orders when possible, if you cannot specify a location or second unit, give a cardinal direction (N, S, E, W, NE, NW, SE, SW)."
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