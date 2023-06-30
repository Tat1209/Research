import time

stopwatch_time_func = time.perf_counter
# stopwatch_time_func = time.time
stopwatch_display_string = "---------------------------------------------------------------------------------------------------  "
# stopwatch_display_string = "-------------  "


stopwatch_time_display = 0.0
stopwatch_time_buffer = None
stopwatch_counter = 0

def start(tag=None, p=True):
    global stopwatch_time_buffer
    if p:
        if tag is None: tag = "start"
        print(stopwatch_display_string + tag)
    stopwatch_time_buffer = stopwatch_time_func()
    
def stop(tag=None):
    add()
    global stopwatch_time_display
    global stopwatch_time_buffer
    if tag is None: tag = "time"
    stopwatch_time_buffer = None
    print(stopwatch_display_string + tag + " : " + str(stopwatch_time_display))

def reset():
    global stopwatch_time_display
    global stopwatch_time_buffer
    global stopwatch_counter
    stopwatch_time_display = 0.0
    stopwatch_time_buffer = None
    stopwatch_counter = 0
    
def lap(tag=None, cnt=False):
    laptime = str(add())
    global stopwatch_time_display
    if tag is None: tag = "lap"
    if cnt:
        global stopwatch_counter
        stopwatch_counter += 1
        tag += str(stopwatch_counter)
    print(stopwatch_display_string + tag + " : " + laptime + "    " + "time : " + str(stopwatch_time_display))
    start(p=False)
    
def add():
    global stopwatch_time_buffer
    diff = stopwatch_time_func() - stopwatch_time_buffer
    global stopwatch_time_display
    stopwatch_time_buffer = None
    stopwatch_time_display += diff
    return diff
    
    
    
    
    
    
    
     
