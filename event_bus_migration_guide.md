# Event Bus Migration Guide

This guide shows exactly how to convert your existing callback-based framework to use an Event Bus pattern.

## Overview

**Before:** FrameProcessor directly calls callbacks
```python
self._safe_call(self.data_to_data_manager_callback, self.most_recent_results)
```

**After:** FrameProcessor publishes events via EventBus
```python
self.event_bus.publish(Event(EventType.PROCESSING_COMPLETE, source="frame_processor", data=self.most_recent_results))
```

---

## Step 1: Create Event Bus Infrastructure

### Create `event_bus.py`:

```python
# event_bus.py
import time
from enum import Enum
from typing import Callable, Dict, List, Any
from dataclasses import dataclass, field
import logging

class EventType(Enum):
    FRAME_READY = "frame_ready"
    PROCESSING_COMPLETE = "processing_complete"
    IMAGE_READY = "image_ready"
    SCAN_ERROR = "scan_error"
    SETTINGS_UPDATED = "settings_updated"
    TUNING_PARAMETERS_UPDATED = "tuning_parameters_updated"
    SYSTEM_ERROR = "system_error"

@dataclass
class Event:
    event_type: EventType
    source: str
    data: Any
    timestamp: float = field(default_factory=time.time)

class EventBus:
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def subscribe(self, event_type: EventType, handler: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        self._logger.debug(f"Subscribed {handler.__name__} to {event_type.value}")
    
    def publish(self, event: Event):
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                self._logger.error(f"Error in handler {handler.__name__}: {e}", exc_info=True)
```

---

## Step 2: Modify `frame_processor_class.py`

### Change 1: Add EventBus to `__init__`

**Find this:**
```python
class FrameProcessor:
    def __init__(self, input_params):
        # ... existing code ...
        self.data_to_data_manager_callback = None
        self.image_to_gui_callback = None
        self.gui_error_callback = None
```

**Replace with:**
```python
class FrameProcessor:
    def __init__(self, input_params, event_bus=None):
        # ... existing code ...
        self.event_bus = event_bus
        # Keep callbacks for backward compatibility during migration
        self.data_to_data_manager_callback = None  # Remove later
        self.image_to_gui_callback = None          # Remove later
        self.gui_error_callback = None             # Remove later
```

### Change 2: Add import at top of file

```python
from event_bus import EventBus, Event, EventType
```

### Change 3: Modify `run_frame_processor()` to publish events

**Find this code in `run_frame_processor()` (around line 789-794):**
```python
if scan_error is None:
    self._safe_call(self.data_to_data_manager_callback, self.most_recent_results)
    # self._safe_call(self.data_to_server_callback, self.most_recent_results)
    self._safe_call(self.image_to_gui_callback, callback_img)
else:
    self._safe_call(self.gui_error_callback, 'SCAN ERROR: ' + scan_error_message)
```

**Replace with:**
```python
if scan_error is None:
    # Publish processing complete event (NEW WAY)
    if self.event_bus:
        processing_event = Event(
            event_type=EventType.PROCESSING_COMPLETE,
            source="frame_processor",
            data=self.most_recent_results
        )
        self.event_bus.publish(processing_event)
        
        image_event = Event(
            event_type=EventType.IMAGE_READY,
            source="frame_processor",
            data=callback_img
        )
        self.event_bus.publish(image_event)
    
    # Keep callback for backward compatibility (OLD WAY - remove later)
    self._safe_call(self.data_to_data_manager_callback, self.most_recent_results)
    self._safe_call(self.image_to_gui_callback, callback_img)
else:
    # Publish error event (NEW WAY)
    if self.event_bus:
        error_event = Event(
            event_type=EventType.SCAN_ERROR,
            source="frame_processor",
            data='SCAN ERROR: ' + scan_error_message
        )
        self.event_bus.publish(error_event)
    
    # Keep callback for backward compatibility (OLD WAY - remove later)
    self._safe_call(self.gui_error_callback, 'SCAN ERROR: ' + scan_error_message)
```

**Note:** During migration, keep both patterns. Once everything works with events, remove the callback lines.

---

## Step 3: Modify `data_manager_class.py`

### Change 1: Add EventBus to `__init__` and subscribe

**Find this:**
```python
class DataManager:
    def __init__(self, params):
        self.params = params['data_frame_parameters']
        # ... rest of init ...
```

**Replace with:**
```python
from event_bus import EventBus, Event, EventType

class DataManager:
    def __init__(self, params, event_bus=None):
        self.params = params['data_frame_parameters']
        self.event_bus = event_bus
        
        # Subscribe to processing complete events
        if self.event_bus:
            self.event_bus.subscribe(
                EventType.PROCESSING_COMPLETE,
                self._handle_processing_complete
            )
        # ... rest of init ...
```

### Change 2: Add event handler method

**Add this new method:**
```python
def _handle_processing_complete(self, event: Event):
    """
    Handle PROCESSING_COMPLETE events from FrameProcessor.
    Replaces the old callback pattern.
    """
    row_data = event.data  # Extract the results dictionary
    print(f'DataManager received data via event: {row_data}')
    
    # Call existing method (no need to duplicate logic)
    self.receive_frame_processor_data(row_data)
```

**Note:** The existing `receive_frame_processor_data()` method doesn't need to change - the event handler just calls it with the data from the event.

---

## Step 4: Modify `gui_manager_class.py`

### Change 1: Add EventBus to `__init__` and subscribe

**Find this:**
```python
class GUIManager:
    def __init__(self, root, params):
        self.root = root
        self.params = params['gui_manager_parameters']
        # ... rest of init ...
```

**Replace with:**
```python
from event_bus import EventBus, Event, EventType

class GUIManager:
    def __init__(self, root, params, event_bus=None):
        self.root = root
        self.params = params['gui_manager_parameters']
        self.event_bus = event_bus
        
        # Subscribe to events
        if self.event_bus:
            self.event_bus.subscribe(
                EventType.IMAGE_READY,
                self._handle_image_ready
            )
            self.event_bus.subscribe(
                EventType.SCAN_ERROR,
                self._handle_scan_error
            )
        # ... rest of init ...
```

### Change 2: Add event handler methods

**Add these new methods:**
```python
def _handle_image_ready(self, event: Event):
    """
    Handle IMAGE_READY events from FrameProcessor.
    Replaces the old receive_img() callback pattern.
    """
    image = event.data
    self.receive_img(image)

def _handle_scan_error(self, event: Event):
    """
    Handle SCAN_ERROR events from FrameProcessor.
    Replaces the old receive_error_callback() pattern.
    """
    error_message = event.data
    self.raise_error(error_message)
```

**Note:** The existing `receive_img()` and `raise_error()` methods don't need to change - the event handlers just call them.

---

## Step 5: Modify `command_center.py`

### Change 1: Create EventBus and pass to components

**Find this in `__init__`:**
```python
class MonocleCommandCenter:
    def __init__(self, camera_parameters):
        # Initialize components (CameraClass, FrameProcessor, etc.)
        self.camera_parameters = camera_parameters
        self.root = tk.Tk()
        self.camera = CameraClass(camera_parameters)
        self.frame_processor = FrameProcessor(camera_parameters)
        self.data_manager = DataManager(camera_parameters)
        self.gui = GUIManager(self.root, camera_parameters)
```

**Replace with:**
```python
from event_bus import EventBus

class MonocleCommandCenter:
    def __init__(self, camera_parameters):
        # Create Event Bus (shared by all components)
        self.event_bus = EventBus()
        
        # Initialize components (CameraClass, FrameProcessor, etc.)
        self.camera_parameters = camera_parameters
        self.root = tk.Tk()
        self.camera = CameraClass(camera_parameters)
        
        # Pass event_bus to components that need it
        self.frame_processor = FrameProcessor(
            camera_parameters,
            event_bus=self.event_bus
        )
        self.data_manager = DataManager(
            camera_parameters,
            event_bus=self.event_bus
        )
        self.gui = GUIManager(
            self.root,
            camera_parameters,
            event_bus=self.event_bus
        )
```

### Change 2: Remove callback setup (optional - can keep during migration)

**Find `initialize_CS_callbacks()` and remove these lines (around line 60-67):**
```python
# Frame Processor sends back processed data and image, or error message when processing is performed
self.frame_processor.establish_data_to_data_manager_callback(
    self.data_manager.receive_frame_processor_data
)
# Frame processor will send result frame to GUI
self.frame_processor.establish_image_to_gui_callback(self.gui.receive_img)
# Frame Processor will send error code to gui
self.frame_processor.establish_gui_error_callback(self.gui.receive_error_callback)
```

**Replace with a comment:**
```python
# These are now handled by Event Bus subscriptions in component __init__ methods:
# - FrameProcessor -> DataManager: PROCESSING_COMPLETE event
# - FrameProcessor -> GUI: IMAGE_READY event
# - FrameProcessor -> GUI: SCAN_ERROR event
```

**Keep these callbacks (they're request/response patterns):**
```python
# GUI sends tuning parameter updates to the frame processor
self.gui.initiate_tuning_parameter_callback(
    self.frame_processor.update_tuning_parameters
)

# GUI sends back settings data when 'Apply Settings' button is pressed
self.gui.callback_settings_params(
    self.frame_processor.receive_gui_settings_data
)

# GUI will send request to Data Manager
self.gui.establish_hist_spinbox_callback(
    self.data_manager.send_requested_data_list
)

# GUI buttons trigger actions
self.gui.establish_start_button_callback(self.start_scan)
self.gui.establish_stop_button_callback(self.stop_processing)
```

---

## Step 6: Test the Migration

### Test 1: Verify Events Are Published

Add logging to see events:
```python
# In event_bus.py, modify publish() to log:
def publish(self, event: Event):
    handlers = self._subscribers.get(event.event_type, [])
    print(f"📢 Publishing {event.event_type.value} from {event.source} to {len(handlers)} handlers")
    for handler in handlers:
        try:
            handler(event)
            print(f"  ✓ Handler {handler.__name__} executed successfully")
        except Exception as e:
            print(f"  ✗ Handler {handler.__name__} failed: {e}")
```

### Test 2: Run Your Application

1. Start the application
2. Trigger a scan
3. Check console for event publishing messages
4. Verify DataManager receives data
5. Verify GUI receives images

### Test 3: Remove Callback Code (Final Step)

Once everything works:

1. **In `frame_processor_class.py`:**
   - Remove `self.data_to_data_manager_callback` attribute
   - Remove `self.image_to_gui_callback` attribute
   - Remove `self.gui_error_callback` attribute
   - Remove callback method: `establish_data_to_data_manager_callback()`
   - Remove callback method: `establish_image_to_gui_callback()`
   - Remove callback method: `establish_gui_error_callback()`
   - Remove callback invocations in `run_frame_processor()`

2. **In `command_center.py`:**
   - Remove callback setup lines from `initialize_CS_callbacks()`

---

## Migration Checklist

- [ ] Create `event_bus.py` file
- [ ] Add EventBus import to `frame_processor_class.py`
- [ ] Modify FrameProcessor `__init__` to accept event_bus
- [ ] Modify FrameProcessor `run_frame_processor()` to publish events
- [ ] Add EventBus import to `data_manager_class.py`
- [ ] Modify DataManager `__init__` to accept event_bus and subscribe
- [ ] Add `_handle_processing_complete()` method to DataManager
- [ ] Add EventBus import to `gui_manager_class.py`
- [ ] Modify GUIManager `__init__` to accept event_bus and subscribe
- [ ] Add `_handle_image_ready()` and `_handle_scan_error()` methods to GUIManager
- [ ] Modify CommandCenter `__init__` to create EventBus and pass to components
- [ ] Test that events are published and received
- [ ] Remove old callback code once verified

---

## Benefits After Migration

✅ **Loose Coupling**: FrameProcessor doesn't know about DataManager or GUI
✅ **Easy to Extend**: Want to log all processing results? Just subscribe to `PROCESSING_COMPLETE`
✅ **Better Error Handling**: One failed handler doesn't crash others
✅ **Easier Testing**: Mock the EventBus instead of all callbacks
✅ **Observability**: Can log/monitor all events in one place

---

## Example: Adding a New Subscriber

**Before (hard):**
1. Modify FrameProcessor to add callback
2. Modify CommandCenter to wire callback
3. Risk breaking existing code

**After (easy):**
```python
class LoggingComponent:
    def __init__(self, event_bus):
        event_bus.subscribe(EventType.PROCESSING_COMPLETE, self.log_results)
    
    def log_results(self, event):
        print(f"Logged: {event.data}")
```

Just subscribe and you're done! No changes needed to FrameProcessor or CommandCenter.
