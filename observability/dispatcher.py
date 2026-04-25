from .tracing.event_schema import get_emitter, Event, EventType
from .metrics.store import get_global_store
from .cli.printer import get_printer
from typing import Dict, Any

class EventDispatcher:
    """Consumes events and routes them to appropriate observability components."""
    
    def __init__(self):
        self.store = get_global_store()
        self.printer = get_printer()
        self._metrics_buffer: Dict[int, Dict[str, Any]] = {}
        self._performance_buffer: Dict[int, Dict[str, float]] = {}

    def handle_event(self, event: Event):
        """Main entry point for handling an event."""
        if event.type == EventType.METRIC:
            self._handle_metric(event)
        elif event.type == EventType.NODE_EXIT:
            self._handle_performance(event)
        elif event.type in [EventType.COMPILER_START, EventType.COMPILER_END]:
            self._handle_trace(event)
        # Add more handlers as needed

    def _handle_performance(self, event: Event):
        # We need a way to know the current step for performance tracking
        # If not provided in event, use a fallback or skip
        step = event.step or 0 # Placeholder
        if step not in self._performance_buffer:
            self._performance_buffer[step] = {}
        
        node_id = event.metadata.get("node_id", event.name)
        self._performance_buffer[step][node_id] = event.duration


    def _handle_metric(self, event: Event):
        # Log to store
        self.store.log(event.name, event.value, step=event.step, metadata=event.metadata)
        
        # Buffer for CLI printing
        if event.step is not None:
            if event.step not in self._metrics_buffer:
                self._metrics_buffer[event.step] = {}
            self._metrics_buffer[event.step][event.name] = event.value
            
            # Every N steps, compute rates and print
            # In a real system, we might check a 'heartbeat' or 'log_frequency'
            if event.name == "loss" and event.step % 100 == 0:
                # Compute rates
                # Note: We need learner_step too, for now let's assume it's in metadata or similar
                # or just use actor_step as a proxy for now.
                rates = self.store.compute_rates(event.step, event.step // 4) # Proxy
                
                metrics_to_print = self._metrics_buffer[event.step]
                metrics_to_print["SPS"] = rates["sps"]
                metrics_to_print["UPS"] = rates["ups"]
                
                self.printer.print_metrics(event.step, metrics_to_print, title="Agent Metrics")
                
                # Clear buffer for this step to save memory
                del self._metrics_buffer[event.step]

        if event.name == "episode_return":
             self.printer.print_panel(
                f"Step {event.step} | Return: {event.value:.2f}",
                title="Episode Finished",
                style="green"
            )



    def _handle_trace(self, event: Event):
        # For now, just print or store traces
        if event.type == EventType.COMPILER_END:
            self.printer.print_panel(
                f"Compilation Finished\nDuration: {event.duration:.4f}s\nNodes: {event.metadata.get('nodes')}",
                title="Compiler",
                style="bold blue"
            )

def setup_default_observability():
    """Wire up the global emitter to the default dispatcher."""
    dispatcher = EventDispatcher()
    emitter = get_emitter()
    emitter.subscribe(dispatcher.handle_event)

# Automatically setup if imported? 
# Or let the user call it. Given this is a library, explicit is better.
