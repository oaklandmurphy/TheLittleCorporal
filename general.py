import ollama
import json

class General:
	def __init__(self, faction: str, identity_prompt, unit_list=None):
		"""
		llm_command: List[str] - The command to run the local LLM (e.g., ["ollama", "run", "mymodel"])
		"""
		self.llm_command = ["ollama", "run", "llama3"]
		self.name = identity_prompt.get("name", "General")
		self.description = identity_prompt.get("description", "")
		self.unit_list = unit_list
		self.unit_summary = self.update_unit_summary()

	def get_instructions(self, player_instructions="", map_summary=""):
		"""
		Passes player instructions and map summary to the LLM and returns the LLM's response as a general.
		"""
		prompt = self._build_prompt(player_instructions, map_summary)
		generals_orders = self._query_general(prompt)
	
		return response

	def _build_prompt(self, player_instructions, map_summary):
		"""
		Builds a prompt for the LLM to act as a battlefield general.
		"""
		prompt = (
			f"You are {self.name}\n"
			f"{self.description}\n"
			"You command the following units:\n"
			f"{self.unit_summary}\n"
			"Given the following battlefield summary and orders from your commanding officer, respond with clear, concise orders for your troops.\n"
			f"Battlefield Summary:\n{map_summary}\n"
			f"Your orders are: {player_instructions}\n"
			"Your response should be in the form of direct orders to each of your units and nothing else."
		)
		print(  prompt  )
		return prompt

	def update_unit_summary(self):
		"""
		Generates a summary of the general's units.
		"""
		if not self.unit_list:
			return "No units assigned."
		summaries = []
		for unit in self.unit_list:
			summaries.append(f"{unit.status()}\n")
		return "\n".join(summaries)

	def _query_general(self, prompt, num_thread=4, num_ctx=4096):
		"""
		Sends the prompt to the local LLM using the ollama Python library and returns its response.
		Limits CPU and RAM usage by setting num_thread and num_ctx.
		"""
		try:
			response = ollama.chat(
				model="llama3",
				messages=[{"role": "user", "content": prompt}],
				options={"num_thread": num_thread, "num_ctx": num_ctx}
			)
			return response["message"]["content"].strip()
		except Exception as e:
			return f"[LLM Error] {e}"
		
	def _staff_officer_prompt(self):
		"""
		Generates a prompt for the staff officer assisting the general.
		"""
		prompt = (
			f"You are the staff officer assisting {self.name}.\n"
			"Your role is to help summarize battlefield information and provide strategic advice.\n"
			"Keep your responses concise and focused on actionable insights."
		)
		return prompt