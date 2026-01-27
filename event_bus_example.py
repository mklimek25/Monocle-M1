"""
Example: How to publish results using Event Bus Pattern
This shows how to convert from callbacks to Event Bus for publishing processing results
"""

import time
from enum import Enum
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

# ============================================================================
# Step 1: Define Event Types
# ============================================================================

class EventType(Enum):
    """All event types in the system."""
    # Camera events
    FRAME_READY = "frame_ready"
    
    # Frame Processor events
    PROCESSING_COMPLETE = "processing_complete"
    IMAGE_READY = "image_ready"
    SCAN_ERROR = "scan_error"
    
    # Data Manager events (if needed later)
    DATA_SAVED = "data_saved"
    
    # GUI events
    SETTINGS_UPDATED = "settings_updated"
    TUNING_PARAMETERS_UPDATED = "tuning_parameters_updated"
    
    # System events
    SYSTEM_ERROR = "system_error"
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"


# ============================================================================
# Step 2: Define Event Data Structure
# ============================================================================

@dataclass
class Event:
    """Represents an event in the system."""
    event_type: EventType
    source: str  # Component that published the event (e.g., "frame_processor")
    data: Any    # Payload specific to event type
    timestamp: float = field(default_factory=time.time)
    
    def __repr__(self):
        return f"Event({self.event_type.value}, source={self.source}, timestamp={self.timestamp})"


# ============================================================================
# Step 3: Implement Event Bus
# ============================================================================

class EventBus:
    """Central event bus for publish/subscribe communication."""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """
        Register a handler for an event type.
        
        Args:
            event_type: The type of event to subscribe to
            handler: Callable that receives Event objects: handler(event: Event) -> None
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            self._logger.debug(f"Subscribed {handler.__name__} to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """Remove a handler from an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)
    
    def publish(self, event: Event):
        """
        Publish an event to all subscribers.
        
        Args:
            event: The event to publish
        """
        handlers = self._subscribers.get(event.event_type, [])
        
        if not handlers:
            self._logger.debug(f"No subscribers for {event.event_type.value}")
            return
        
        self._logger.debug(f"Publishing {event} to {len(handlers)} subscribers")
        
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Log error but don't crash - one failed handler shouldn't stop others
                self._logger.error(
                    f"Error in event handler {handler.__name__} for {event.event_type.value}: {e}",
                    exc_info=True
                )


# ============================================================================
# Step 4: Modify FrameProcessor to Publish Events
# ============================================================================

class FrameProcessor:
    """
    Example of how FrameProcessor would publish events instead of using callbacks.
    This replaces the callback pattern with event publishing.
    """
    
    def __init__(self, input_params, event_bus: Optional[EventBus] = None):
        """Initialize with optional EventBus (injected via DI)."""
        self.general_params = input_params['general_params']
        self.frame_processor_params = input_params['frame_proccessor_params']
        
        # Store event bus reference
        self.event_bus = event_bus
        
        # Remove old callback attributes (these get replaced)
        # self.data_to_data_manager_callback = None  # OLD WAY
        # self.image_to_gui_callback = None          # OLD WAY
        # self.gui_error_callback = None             # OLD WAY
        
        # ... rest of initialization ...
        self.most_recent_results = None
        self.stop_event = None
        # ... etc ...
    
    def run_frame_processor(self):
        """Continuously process frames from the queue."""
        # ... existing processing code ...
        
        # Example processing loop (simplified)
        while not self.stop_event.is_set():
            # ... process frames ...
            
            # After processing completes, you have results:
            height_results = [1.5, 1.6, 1.7, 1.8, 1.9]
            average_height = 1.7
            width_results = [2.0, 2.1, 2.2]
            average_width = 2.1
            area_result = 3.5
            yield_result = 0.85
            scan_error = None
            product_error = None
            callback_img = None  # Generated image
            
            self.most_recent_results = {
                'timestamp': time.time(),
                'product': 'product_name',
                'reading_number': 1,
                'height_results': height_results,
                'average_height': average_height,
                'width_results': width_results,
                'average_width': average_width,
                'area_val': area_result,
                'ret_yield': yield_result,
                'scan_error': scan_error,
                'product_error': product_error
            }
            
            # ====================================================================
            # OLD WAY: Direct callback invocation
            # ====================================================================
            # if scan_error is None:
            #     self._safe_call(self.data_to_data_manager_callback, self.most_recent_results)
            #     self._safe_call(self.image_to_gui_callback, callback_img)
            # else:
            #     self._safe_call(self.gui_error_callback, 'SCAN ERROR: ' + scan_error_message)
            
            # ====================================================================
            # NEW WAY: Publish events via Event Bus
            # ====================================================================
            if self.event_bus is None:
                raise ValueError("EventBus not provided to FrameProcessor")
            
            if scan_error is None:
                # Publish processing complete event (DataManager will subscribe)
                processing_event = Event(
                    event_type=EventType.PROCESSING_COMPLETE,
                    source="frame_processor",
                    data=self.most_recent_results
                )
                self.event_bus.publish(processing_event)
                
                # Publish image ready event (GUI will subscribe)
                image_event = Event(
                    event_type=EventType.IMAGE_READY,
                    source="frame_processor",
                    data=callback_img
                )
                self.event_bus.publish(image_event)
                
            else:
                # Publish error event (GUI will subscribe)
                error_event = Event(
                    event_type=EventType.SCAN_ERROR,
                    source="frame_processor",
                    data=f'SCAN ERROR: {scan_error_message}'
                )
                self.event_bus.publish(error_event)


# ============================================================================
# Step 5: Modify DataManager to Subscribe to Events
# ============================================================================

class DataManager:
    """
    Example of how DataManager would subscribe to events instead of receiving callbacks.
    """
    
    def __init__(self, params, event_bus: Optional[EventBus] = None):
        self.params = params['data_frame_parameters']
        
        # Store event bus reference
        self.event_bus = event_bus
        
        # Subscribe to events if event bus provided
        if self.event_bus:
            self.event_bus.subscribe(
                EventType.PROCESSING_COMPLETE,
                self._handle_processing_complete
            )
    
    def _handle_processing_complete(self, event: Event):
        """
        Handler for PROCESSING_COMPLETE events.
        This replaces the old callback: receive_frame_processor_data()
        
        Args:
            event: Event containing processing results in event.data
        """
        row_data = event.data  # This is the same dict that was passed via callback
        
        print(f'DataManager received processing results via event: {row_data}')
        
        # Use existing method logic
        self.receive_frame_processor_data(row_data)
    
    def receive_frame_processor_data(self, row_data):
        """
        Existing method - kept as-is, now called by event handler.
        This maintains backward compatibility and isolates the data processing logic.
        """
        # ... existing implementation ...
        print(f'Processing data: {row_data}')


# ============================================================================
# Step 6: Modify GUIManager to Subscribe to Events
# ============================================================================

class GUIManager:
    """
    Example of how GUIManager would subscribe to events instead of receiving callbacks.
    """
    
    def __init__(self, root, params, event_bus: Optional[EventBus] = None):
        self.root = root
        self.params = params['gui_manager_parameters']
        
        # Store event bus reference
        self.event_bus = event_bus
        
        # Subscribe to events if event bus provided
        if self.event_bus:
            self.event_bus.subscribe(
                EventType.IMAGE_READY,
                self._handle_image_ready
            )
            self.event_bus.subscribe(
                EventType.SCAN_ERROR,
                self._handle_scan_error
            )
            self.event_bus.subscribe(
                EventType.SYSTEM_ERROR,
                self._handle_system_error
            )
    
    def _handle_image_ready(self, event: Event):
        """
        Handler for IMAGE_READY events.
        Replaces old callback: receive_img()
        """
        image = event.data  # This is the same image that was passed via callback
        self.receive_img(image)
    
    def _handle_scan_error(self, event: Event):
        """
        Handler for SCAN_ERROR events.
        Replaces old callback: receive_error_callback()
        """
        error_message = event.data  # This is the same error message
        self.raise_error(error_message)
    
    def _handle_system_error(self, event: Event):
        """Handler for system-level errors."""
        error_message = event.data
        self.raise_error(f"System Error: {error_message}")
    
    def receive_img(self, image):
        """Existing method - kept as-is, now called by event handler."""
        # ... existing implementation ...
        print(f'GUI received image: {image.shape if hasattr(image, "shape") else "N/A"}')
    
    def raise_error(self, text):
        """Existing method - kept as-is, now called by event handler."""
        # ... existing implementation ...
        print(f'GUI Error: {text}')


# ============================================================================
# Step 7: Wire Everything Together in CommandCenter
# ============================================================================

class MonocleCommandCenter:
    """
    Refactored CommandCenter showing Event Bus integration.
    """
    
    def __init__(self, camera_parameters):
        # Create the Event Bus (single instance shared by all components)
        self.event_bus = EventBus()
        
        # Initialize components and inject event bus
        self.camera_parameters = camera_parameters
        self.root = tk.Tk()
        
        # Pass event_bus to components that need to publish/subscribe
        # Note: Camera might not need it if it still uses callbacks for frames
        self.camera = CameraClass(camera_parameters)
        
        # FrameProcessor needs event_bus to PUBLISH events
        self.frame_processor = FrameProcessor(
            camera_parameters,
            event_bus=self.event_bus
        )
        
        # DataManager needs event_bus to SUBSCRIBE to events
        self.data_manager = DataManager(
            camera_parameters,
            event_bus=self.event_bus
        )
        
        # GUIManager needs event_bus to SUBSCRIBE to events
        self.gui = GUIManager(
            self.root,
            camera_parameters,
            event_bus=self.event_bus
        )
        
        # Setup remaining connections
        # Note: Camera -> FrameProcessor might still use callback (or convert to event)
        self.camera.establish_frame_to_processor_callback(
            self.frame_processor.queue_frame_processor
        )
        
        # GUI still needs to send commands (settings, tuning params)
        # These could also be converted to events if desired
        self.gui.initiate_tuning_parameter_callback(
            self.frame_processor.update_tuning_parameters
        )
        
        # ... other setup ...
    
    def initialize_CS_callbacks(self):
        """
        Simplified version - most callbacks replaced by Event Bus.
        Only keep callbacks for immediate request/response patterns.
        """
        # Camera sends frames to processor (might keep as callback for speed)
        self.camera.establish_frame_to_processor_callback(
            self.frame_processor.queue_frame_processor
        )
        
        # GUI sends tuning parameters (bidirectional request/response - could stay as callback)
        self.gui.initiate_tuning_parameter_callback(
            self.frame_processor.update_tuning_parameters
        )
        
        # GUI sends settings (bidirectional - could stay as callback)
        self.gui.callback_settings_params(
            self.frame_processor.receive_gui_settings_data
        )
        
        # GUI requests data from DataManager (bidirectional request/response)
        self.gui.establish_hist_spinbox_callback(
            self.data_manager.send_requested_data_list
        )
        
        # GUI buttons trigger actions
        self.gui.establish_start_button_callback(self.start_scan)
        self.gui.establish_stop_button_callback(self.stop_processing)
        
        # NOTE: These are now handled by Event Bus subscriptions:
        # - FrameProcessor -> DataManager (PROCESSING_COMPLETE event)
        # - FrameProcessor -> GUI (IMAGE_READY event)
        # - FrameProcessor -> GUI (SCAN_ERROR event)
    
    def start_scan(self, boolean):
        """Start scanning - can publish events too."""
        if boolean:
            # Publish system started event
            self.event_bus.publish(Event(
                event_type=EventType.SYSTEM_STARTED,
                source="command_center",
                data={"started": True}
            ))
            # ... rest of start logic ...


# ============================================================================
# COMPARISON: Old vs New Pattern
# ============================================================================

def comparison_example():
    """
    Side-by-side comparison of old callback pattern vs new event bus pattern.
    """
    
    # ========================================================================
    # OLD WAY: Callbacks
    # ========================================================================
    print("=" * 70)
    print("OLD WAY: Direct Callbacks")
    print("=" * 70)
    
    # Setup (in CommandCenter.__init__)
    # frame_processor.data_to_data_manager_callback = data_manager.receive_frame_processor_data
    # frame_processor.image_to_gui_callback = gui.receive_img
    # frame_processor.gui_error_callback = gui.raise_error
    
    # Publishing (in FrameProcessor.run_frame_processor)
    # if scan_error is None:
    #     self._safe_call(self.data_to_data_manager_callback, self.most_recent_results)
    #     self._safe_call(self.image_to_gui_callback, callback_img)
    # else:
    #     self._safe_call(self.gui_error_callback, 'SCAN ERROR: ' + scan_error_message)
    
    # Problems:
    # - Tight coupling: FrameProcessor must know about DataManager and GUI
    # - Hard to add new subscribers (e.g., logging component)
    # - Manual callback setup required
    # - Error in one callback can break others
    
    print("\n")
    
    # ========================================================================
    # NEW WAY: Event Bus
    # ========================================================================
    print("=" * 70)
    print("NEW WAY: Event Bus")
    print("=" * 70)
    
    # Setup (automatic via subscription in DataManager/GUI.__init__)
    # data_manager subscribes: event_bus.subscribe(PROCESSING_COMPLETE, handler)
    # gui subscribes: event_bus.subscribe(IMAGE_READY, handler)
    # gui subscribes: event_bus.subscribe(SCAN_ERROR, handler)
    
    # Publishing (in FrameProcessor.run_frame_processor)
    # if scan_error is None:
    #     event_bus.publish(Event(PROCESSING_COMPLETE, data=results))
    #     event_bus.publish(Event(IMAGE_READY, data=image))
    # else:
    #     event_bus.publish(Event(SCAN_ERROR, data=error_message))
    
    # Benefits:
    # - Loose coupling: FrameProcessor doesn't know who subscribes
    # - Easy to add subscribers (just subscribe, no changes to publisher)
    # - Automatic delivery to all subscribers
    # - Errors in one handler don't affect others
    # - Can log/monitor all events centrally


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    import tkinter as tk
    from camera_parameters import monocle_parameters
    
    # Setup logging to see event flow
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create command center (it will create EventBus internally)
    # command_center = MonocleCommandCenter(monocle_parameters)
    # command_center.initialize_CS_callbacks()
    # command_center.run_gui()
    
    print("\n" + "=" * 70)
    print("Event Bus Example Complete")
    print("=" * 70)
    print("\nKey Points:")
    print("1. EventBus is created once in CommandCenter")
    print("2. Components receive EventBus via dependency injection")
    print("3. Publishers call: event_bus.publish(Event(...))")
    print("4. Subscribers call: event_bus.subscribe(EventType, handler)")
    print("5. Event handlers receive Event objects with .data containing payload")
    print("\nNext Steps:")
    print("- Implement EventBus class in your project")
    print("- Modify FrameProcessor.run_frame_processor() to publish events")
    print("- Modify DataManager.__init__() to subscribe to events")
    print("- Modify GUIManager.__init__() to subscribe to events")
    print("- Remove old callback attributes and setup code")
