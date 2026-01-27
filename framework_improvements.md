# Framework Improvement Recommendations for command_center.py

## Executive Summary

The current `MonocleCommandCenter` uses callback-based communication with tight coupling. This document outlines architectural improvements for better maintainability, testability, and extensibility.

---

## 1. **Dependency Injection & Service Locator Pattern**

### Current Problem:
- Direct instantiation of components in `__init__`
- Hard to test (can't mock dependencies)
- Tight coupling between components

### Recommended Solution:

```python
class MonocleCommandCenter:
    def __init__(self, camera_parameters, 
                 camera=None, frame_processor=None, 
                 data_manager=None, gui_manager=None):
        """
        Use dependency injection for better testability.
        Components can be passed in (for testing) or created internally.
        """
        self.camera_parameters = camera_parameters
        
        # Use provided instances or create new ones
        self.camera = camera or CameraClass(camera_parameters)
        self.frame_processor = frame_processor or FrameProcessor(camera_parameters)
        self.data_manager = data_manager or DataManager(camera_parameters)
        
        self.root = tk.Tk()
        self.gui = gui_manager or GUIManager(self.root, camera_parameters)
```

**Benefits:**
- Easy to unit test with mock objects
- Can swap implementations (e.g., mock camera for testing)
- Better separation of concerns

---

## 2. **Event Bus / Message Broker Pattern**

### Current Problem:
- Manual callback setup in `initialize_CS_callbacks()` (40+ lines)
- Callbacks are tightly coupled (component A directly knows about component B)
- Adding new communication paths requires modifying multiple files
- Error-prone: easy to miss setting up a callback

### Recommended Solution:

```python
from enum import Enum
from typing import Callable, Dict, List
from dataclasses import dataclass

class EventType(Enum):
    FRAME_READY = "frame_ready"
    PROCESSING_COMPLETE = "processing_complete"
    DATA_READY = "data_ready"
    ERROR_OCCURRED = "error_occurred"
    SETTINGS_UPDATED = "settings_updated"
    GUI_UPDATE_REQUESTED = "gui_update_requested"

@dataclass
class Event:
    event_type: EventType
    source: str
    data: Any
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class EventBus:
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """Register a handler for an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    def publish(self, event: Event):
        """Publish an event to all subscribers."""
        handlers = self._subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Log error but don't crash the system
                print(f"Error in event handler for {event.event_type}: {e}")

class MonocleCommandCenter:
    def __init__(self, camera_parameters):
        self.event_bus = EventBus()
        # ... initialize components ...
        self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self):
        """Centralized event subscription setup."""
        # Camera publishes frames
        self.event_bus.subscribe(EventType.FRAME_READY, 
                                 self.frame_processor.queue_frame_processor)
        
        # Frame processor publishes results
        self.event_bus.subscribe(EventType.PROCESSING_COMPLETE,
                                 self.data_manager.receive_frame_processor_data)
        
        # Error handling
        self.event_bus.subscribe(EventType.ERROR_OCCURRED,
                                 self.gui.receive_error_callback)
        
        # ... etc ...
```

**Benefits:**
- Loose coupling: components don't know about each other
- Easy to add new event types and subscribers
- Better error handling (one failed handler doesn't crash others)
- Easier to debug (can log all events)

---

## 3. **State Machine for System Lifecycle**

### Current Problem:
- No clear state management
- Hard to know if system is initializing, running, stopping, etc.
- Race conditions possible when starting/stopping

### Recommended Solution:

```python
from enum import Enum
import threading

class SystemState(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

class StateMachine:
    def __init__(self, initial_state=SystemState.INITIALIZING):
        self._state = initial_state
        self._lock = threading.Lock()
        self._state_change_callbacks = []
    
    @property
    def state(self):
        with self._lock:
            return self._state
    
    def transition_to(self, new_state: SystemState, allow_transition: Callable = None):
        """Transition to a new state if allowed."""
        with self._lock:
            if allow_transition and not allow_transition(self._state, new_state):
                raise ValueError(f"Invalid state transition: {self._state} -> {new_state}")
            
            old_state = self._state
            self._state = new_state
            
            # Notify listeners
            for callback in self._state_change_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    print(f"Error in state change callback: {e}")
    
    def on_state_change(self, callback: Callable):
        """Register callback for state changes."""
        self._state_change_callbacks.append(callback)

class MonocleCommandCenter:
    def __init__(self, camera_parameters):
        # ...
        self.state_machine = StateMachine(SystemState.INITIALIZING)
        self.state_machine.transition_to(SystemState.READY)
    
    def start_scan(self, boolean):
        if self.state_machine.state != SystemState.READY:
            self.gui.raise_error("System not ready to start")
            return
        
        self.state_machine.transition_to(SystemState.STARTING)
        # ... start threads ...
        self.state_machine.transition_to(SystemState.RUNNING)
```

**Benefits:**
- Clear system state visibility
- Prevents invalid operations (e.g., starting when already running)
- Better debugging (can query state at any time)

---

## 4. **Thread Pool & Task Queue Pattern**

### Current Problem:
- Manual thread creation and management
- No thread pooling or task prioritization
- Hard to add new concurrent operations

### Recommended Solution:

```python
from concurrent.futures import ThreadPoolExecutor, Future
import queue

class TaskQueue:
    def __init__(self, max_size=100):
        self._queue = queue.PriorityQueue(maxsize=max_size)
    
    def enqueue(self, task, priority=5):
        """Add task to queue. Lower priority = higher priority."""
        self._queue.put((priority, task))
    
    def dequeue(self, timeout=None):
        """Get next task from queue."""
        try:
            priority, task = self._queue.get(timeout=timeout)
            return task
        except queue.Empty:
            return None

class ThreadManager:
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []
    
    def submit_task(self, fn, *args, **kwargs) -> Future:
        """Submit a task and track it."""
        future = self.executor.submit(fn, *args, **kwargs)
        self._futures.append(future)
        return future
    
    def shutdown(self, wait=True, timeout=30):
        """Gracefully shutdown all threads."""
        self.executor.shutdown(wait=wait, timeout=timeout)
```

**Benefits:**
- Better resource management
- Can prioritize tasks
- Easier to scale up/down workers
- Built-in error handling via futures

---

## 5. **Configuration Management**

### Current Problem:
- Parameters passed as nested dictionaries
- No validation
- Hard to override for testing

### Recommended Solution:

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class CameraConfig:
    video_test: bool = False
    video_name: Optional[str] = None
    camera_matrix: Any = None
    # ... other params ...
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CameraConfig':
        """Create config from dictionary with validation."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

@dataclass
class SystemConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    general: GeneralConfig = field(default_factory=GeneralConfig)
    # ... other configs ...
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SystemConfig':
        """Build config from nested dictionary."""
        return cls(
            camera=CameraConfig.from_dict(d.get('camera_params', {})),
            general=GeneralConfig.from_dict(d.get('general_params', {})),
            # ...
        )
```

**Benefits:**
- Type safety and IDE autocomplete
- Validation at config creation
- Clear structure
- Easy to serialize/deserialize

---

## 6. **Error Handling & Recovery Strategy**

### Current Problem:
- Errors sometimes crash threads
- No retry logic
- Limited error recovery

### Recommended Solution:

```python
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def retry_on_failure(max_attempts=3, delay=1.0, backoff=2.0):
    """Decorator for retrying failed operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"{func.__name__} attempt {attempt} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

class CircuitBreaker:
    """Prevent cascade failures."""
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise
```

---

## 7. **Improved Watchdog Pattern**

### Current Problem:
- Watchdog only checks thread liveness
- Doesn't monitor thread health/performance
- No metrics collection

### Recommended Solution:

```python
class ThreadHealthMonitor:
    def __init__(self, check_interval=2.0):
        self.check_interval = check_interval
        self._monitored_threads = {}
        self._stop_event = threading.Event()
        self._thread = None
    
    def register_thread(self, name: str, thread: threading.Thread, 
                       health_check: Callable = None):
        """Register a thread for monitoring."""
        self._monitored_threads[name] = {
            'thread': thread,
            'health_check': health_check,
            'last_check': time.time(),
            'restart_count': 0
        }
    
    def start(self):
        """Start monitoring thread."""
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self._stop_event.is_set():
            for name, info in self._monitored_threads.items():
                thread = info['thread']
                
                # Check if thread is alive
                if not thread.is_alive():
                    logger.warning(f"Thread {name} is not alive. Attempting restart...")
                    self._restart_thread(name, info)
                
                # Run custom health check
                if info['health_check']:
                    try:
                        if not info['health_check']():
                            logger.warning(f"Health check failed for {name}")
                    except Exception as e:
                        logger.error(f"Health check error for {name}: {e}")
            
            self._stop_event.wait(self.check_interval)
    
    def _restart_thread(self, name: str, info: dict):
        """Restart a failed thread."""
        # This would need to know how to recreate the thread
        # Could use a factory function stored in the info dict
        info['restart_count'] += 1
        # ... restart logic ...
```

---

## 8. **Improved Callback Safety**

### Current Problem:
- `_safe_call` exists but is ad-hoc
- No standardized callback interface
- Callbacks can block

### Recommended Solution:

```python
from typing import Protocol
import asyncio

class CallbackProtocol(Protocol):
    """Protocol for callbacks."""
    def __call__(self, *args, **kwargs) -> None:
        ...

class SafeCallback:
    """Wraps callbacks with error handling and optional async execution."""
    
    def __init__(self, callback: Callable, 
                 async_execution: bool = False,
                 timeout: float = None):
        self.callback = callback
        self.async_execution = async_execution
        self.timeout = timeout
    
    def __call__(self, *args, **kwargs):
        if self.async_execution:
            # Execute in background thread to avoid blocking
            threading.Thread(
                target=self._execute_safe,
                args=args,
                kwargs=kwargs,
                daemon=True
            ).start()
        else:
            self._execute_safe(*args, **kwargs)
    
    def _execute_safe(self, *args, **kwargs):
        try:
            if self.timeout:
                # Use signal or threading timeout for Unix/Windows
                result = self.callback(*args, **kwargs)
            else:
                result = self.callback(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Callback {self.callback.__name__} failed: {e}", 
                        exc_info=True)
            # Optionally notify error handler via event bus
```

---

## 9. **Dependency Graph & Startup Sequence**

### Current Problem:
- Startup order is implicit
- Hard to add new components
- No dependency validation

### Recommended Solution:

```python
from typing import List, Set

class ComponentDependencyGraph:
    def __init__(self):
        self._dependencies: Dict[str, Set[str]] = {}
        self._components: Dict[str, Any] = {}
    
    def register(self, name: str, component: Any, depends_on: List[str] = None):
        """Register a component and its dependencies."""
        self._components[name] = component
        self._dependencies[name] = set(depends_on or [])
    
    def get_startup_order(self) -> List[str]:
        """Topological sort to determine startup order."""
        # Simple topological sort implementation
        visited = set()
        result = []
        
        def visit(name):
            if name in visited:
                return
            visited.add(name)
            for dep in self._dependencies.get(name, []):
                visit(dep)
            result.append(name)
        
        for name in self._components:
            visit(name)
        
        return result
    
    def initialize_all(self):
        """Initialize all components in dependency order."""
        order = self.get_startup_order()
        for name in order:
            component = self._components[name]
            if hasattr(component, 'initialize'):
                component.initialize()

class MonocleCommandCenter:
    def __init__(self, camera_parameters):
        # ...
        self.dependency_graph = ComponentDependencyGraph()
        
        # Register components with dependencies
        self.dependency_graph.register("camera", self.camera)
        self.dependency_graph.register("frame_processor", self.frame_processor, 
                                       depends_on=["camera"])
        self.dependency_graph.register("data_manager", self.data_manager)
        self.dependency_graph.register("gui", self.gui, 
                                       depends_on=["data_manager", "frame_processor"])
        
        # Initialize in correct order
        self.dependency_graph.initialize_all()
```

---

## 10. **Logging & Observability**

### Current Problem:
- Print statements instead of proper logging
- No structured logging
- Hard to debug production issues

### Recommended Solution:

```python
import logging
from logging.handlers import RotatingFileHandler
import json

class StructuredLogger:
    def __init__(self, name: str, log_file: str = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler with rotation
        if log_file:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(logging.DEBUG)
            json_formatter = logging.Formatter(
                '%(message)s'  # Assume JSON messages for file
            )
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.addHandler(console_handler)
    
    def log_event(self, event_type: str, **kwargs):
        """Log structured event."""
        self.logger.info(json.dumps({
            'event_type': event_type,
            'timestamp': time.time(),
            **kwargs
        }))
```

---

## Implementation Priority

### Phase 1 (High Impact, Low Risk):
1. **Logging & Observability** - Replace prints with proper logging
2. **Configuration Management** - Use dataclasses for config
3. **Improved Error Handling** - Add retry logic and better exception handling

### Phase 2 (High Impact, Medium Risk):
4. **Event Bus Pattern** - Replace callbacks with event bus
5. **State Machine** - Add lifecycle state management
6. **Safe Callbacks** - Standardize callback execution

### Phase 3 (Medium Impact, Lower Priority):
7. **Thread Pool** - Move to ThreadPoolExecutor
8. **Dependency Injection** - Refactor constructor
9. **Dependency Graph** - Add startup sequencing
10. **Enhanced Watchdog** - Improve monitoring

---

## Example: Refactored Command Center Skeleton

```python
class MonocleCommandCenter:
    def __init__(self, config: SystemConfig,
                 camera: Optional[CameraClass] = None,
                 frame_processor: Optional[FrameProcessor] = None,
                 data_manager: Optional[DataManager] = None,
                 gui_manager: Optional[GUIManager] = None):
        
        # Setup logging
        self.logger = StructuredLogger(self.__class__.__name__)
        
        # Configuration
        self.config = config
        
        # State management
        self.state_machine = StateMachine(SystemState.INITIALIZING)
        
        # Event bus for loose coupling
        self.event_bus = EventBus()
        
        # Dependency injection
        self.camera = camera or CameraClass(config.camera)
        self.frame_processor = frame_processor or FrameProcessor(config.frame_processor)
        self.data_manager = data_manager or DataManager(config.data_manager)
        
        self.root = tk.Tk()
        self.gui = gui_manager or GUIManager(self.root, config.gui)
        
        # Thread management
        self.thread_manager = ThreadManager(max_workers=4)
        
        # Health monitoring
        self.health_monitor = ThreadHealthMonitor()
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        # Transition to ready state
        self.state_machine.transition_to(SystemState.READY)
        self.logger.log_event("system_initialized")
    
    def _setup_event_subscriptions(self):
        """Setup event bus subscriptions."""
        self.event_bus.subscribe(
            EventType.FRAME_READY,
            SafeCallback(self.frame_processor.queue_frame_processor)
        )
        # ... more subscriptions ...
    
    @retry_on_failure(max_attempts=3)
    def start_scan(self, boolean):
        """Start scanning with state validation."""
        if self.state_machine.state != SystemState.READY:
            self.event_bus.publish(Event(
                EventType.ERROR_OCCURRED,
                source="command_center",
                data="System not ready to start"
            ))
            return
        
        self.state_machine.transition_to(SystemState.STARTING)
        # ... start operations ...
        self.state_machine.transition_to(SystemState.RUNNING)
        self.logger.log_event("scan_started")
```

---

## Additional Recommendations

1. **Type Hints**: Add comprehensive type hints throughout for better IDE support
2. **Unit Tests**: Use dependency injection to enable easy mocking
3. **Documentation**: Add docstrings following Google/NumPy style
4. **Async/Await**: Consider using asyncio for I/O-bound operations instead of threads
5. **Configuration File**: Move parameters to YAML/JSON config file instead of Python dict
6. **Plugin Architecture**: Design for extensibility (plugins for different camera types, processors, etc.)

---

## Conclusion

These improvements will make the framework:
- **More Testable**: Dependency injection and event bus enable easy mocking
- **More Maintainable**: Clear separation of concerns and state management
- **More Robust**: Better error handling and recovery strategies
- **More Extensible**: Event bus and dependency graph make adding features easier
- **More Observable**: Proper logging and health monitoring

Start with Phase 1 improvements, then gradually refactor to Phase 2 and 3 patterns.

