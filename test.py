from gpiozero import LED, Buzzer
import time
def setup_gp():
	
	buz = Buzzer(18)
	led = LED(23)

	return buz, led

buz, led = setup_gp()
buz.on()
led.on()
time.sleep(1)
buz.off()
led.off()
