A simple ens domain appraiser app.
ensure that all the needed dependencies are installed. These are found in the requirements.txt file

install dependencies using the following command:
pip install -r requirements.txt

To appraise your domain using the app, send a POST request to IP address of your deployment server.
query takes the form key=domain, value="your domain name" (if you are using postman)

If you are using a cli, you can use the following command:
curl -X POST -d "domain=your domain name" http://your_IP_address

to see if the appraiser has been deployed succesfully going to the address of your deployment server should show the following message
"Appraiser is active"
