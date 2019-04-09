from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait

driver = webdriver.Chrome("E:\\Program\\Anaconda3\\chromedriver.exe")

locator = (By.XPATH, '//*[@id="root"]/div/div[1]/div/div/div/div[1]/div[1]/div/div[2]')


WebDriverWait(driver, 20, 0.5).until(EC.presence_of_element_located(locator))
driver.find_element_by_xpath('//*[@id="root"]/div/div[1]/div/div/div/div[1]/div[1]/div/div[2]').click()
By.CLASS_NAME
driver.navigate().refresh()