from map import Map
from unit import Unit
import pygame
import sys
import queue
import threading
from reporting import generate_tactical_report

class TurnManager:
    """Manages turn order and unit actions for factions."""
    def __init__(self, game_map: Map, factions: list[str]):
        self.map = game_map
        self.factions = factions
        self.current_faction_index = 0

    def get_current_faction(self):
        """Get the name of the current faction."""
        return self.factions[self.current_faction_index]

    def all_units(self):
        """Collect all units currently on the map."""
        units = []
        for row in self.map.grid:
            for hex in row:
                if hex.unit:
                    units.append(hex.unit)
        return units

    def process_turn_start(self):
        """Execute turn start sequence: engagement check, reset mobility, apply combat damage."""
        current_faction = self.get_current_faction()
        
        # 1. Print which faction's turn it is
        print(f"\n{'='*60}")
        print(f"--- {current_faction.upper()} TURN ---")
        print(f"{'='*60}")
        
        # 2. Reset engagement flags from previous turn
        for unit in self.all_units():
            unit.engagement = 0
        
        # 3. Check for all units engagement status then engaged units deal damage to each other
        # Pass all factions to check engagements (assuming 2-faction game)
        if len(self.factions) >= 2:
            self.map.check_all_engagements(self.factions[0], self.factions[1])
        
        # 4. Reset mobility for current faction's units
        for unit in self.all_units():
            unit.reset_mobility()

    def advance_to_next_faction(self):
        """Move to the next faction's turn."""
        self.current_faction_index = (self.current_faction_index + 1) % len(self.factions)

    def run_game_loop(self, vis, generals, clock, max_retries=9, num_threads=4, num_ctx=4096):
        """Main game loop that handles rendering, input, and turn processing."""
        running = True
        
        # Queues and events for multithreading
        prompt_queue = queue.Queue()
        input_allowed = threading.Event()
        llm_result_queue = queue.Queue()
        llm_processing = threading.Event()
        
        # Flags for turn state management
        turn_started = False
        waiting_for_input = False
        
        # Input thread - runs continuously in background
        def input_thread():
            while running:
                input_allowed.wait()
                if not running:
                    break
                try:
                    prompt = input("\nType your orders for the General (or 'quit' to exit): ")
                    prompt_queue.put(prompt)
                    input_allowed.clear()
                except EOFError:
                    break
        
        input_thread_obj = threading.Thread(target=input_thread, daemon=True)
        input_thread_obj.start()
        
        # LLM processing thread - handles general order generation and execution
        def llm_processing_thread(player_prompt, current_faction, general, map_summary_general, game_map):
            """Run LLM calls in background thread to keep pygame responsive."""
            try:
                print(f"\n[{current_faction} General's Turn]")
                
                # General responds to player instructions
                general_response = general.get_instructions(
                    player_instructions=player_prompt, 
                    map_summary=map_summary_general
                )
                print(f"\n[{current_faction} General's Orders]:\n{general_response}")
                
                # Parse the JSON orders
                orders_data = general.parse_orders_json(general_response)
                
                # Execute orders directly on the map
                game_map.execute_orders(orders_data, current_faction)
                
                llm_result_queue.put({
                    "success": True,
                    "applied": True,
                    "current_faction": current_faction
                })
            except Exception as e:
                print(f"[LLM Processing Error] {e}")
                llm_result_queue.put({
                    "success": False,
                    "error": str(e),
                    "current_faction": current_faction
                })
            finally:
                llm_processing.clear()
        
        # Main game loop
        while running:
            # Handle pygame events
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
            
            # TURN SEQUENCE
            # Step 1-4: Process turn start (engagement, mobility, combat damage)
            if not turn_started:
                self.process_turn_start()
                turn_started = True
                waiting_for_input = True
            
            # Step 5: Print battlefield summary and Step 6: Prompt for input
            if waiting_for_input and not llm_processing.is_set():
                current_faction = self.get_current_faction()
                current_general = generals[current_faction]
                
                # Generate and print tactical report
                map_summary_general = generate_tactical_report(
                    self.map, 
                    current_faction, 
                    current_general.unit_list,
                )
                print("\n" + map_summary_general)
                
                # Allow input
                waiting_for_input = False
                input_allowed.set()
            
            # Step 7: Check for player input and resolve LLM interactions
            if not prompt_queue.empty() and not llm_processing.is_set():
                player_prompt = prompt_queue.get()
                
                # Handle quit command
                if player_prompt.strip().lower() == "quit":
                    running = False
                    input_allowed.set()
                    continue
                
                # Get current turn information
                current_faction = self.get_current_faction()
                general = generals[current_faction]
                
                # Start LLM processing in background
                llm_processing.set()
                llm_thread = threading.Thread(
                    target=llm_processing_thread,
                    args=(player_prompt, current_faction, general, map_summary_general, self.map),
                    daemon=True
                )
                llm_thread.start()
            
            # Check if LLM processing has completed
            if not llm_result_queue.empty():
                result = llm_result_queue.get()
                
                if result["success"]:
                    print(f"\n[Turn Complete] Orders executed successfully.")
                    
                    # Advance to next faction's turn
                    self.advance_to_next_faction()
                    turn_started = False
                else:
                    # Error occurred during order processing
                    print(f"\n[Order Processing ERROR] {result.get('error', 'Unknown error')}")
                    print("[Turn skipped - order processing failed]")
                    
                    # Still advance turn even on failure
                    self.advance_to_next_faction()
                    turn_started = False
            
            # Render the visualization
            hover_info = vis.get_hover_info(mouse_pos)
            vis.render(hover_info, llm_processing=llm_processing.is_set())
            pygame.display.flip()
            clock.tick(30)
        
        # Clean up
        pygame.quit()
        sys.exit()
