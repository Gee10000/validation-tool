function sendMail() {
    var name = document.getElementById("name").value;
    var email = document.getElementById("email").value;
    var message = document.getElementById("message").value;

    if (!validateEmail(email)) {
    alert("Please enter a valid email address.");
    return;
    }

  var params = {
    name: name,
    email: email,
    message: message,
  };


 const serviceID = "service_83h8efd";
 const templateID = "template_s4jofnq";

 emailjs
    .send(serviceID, templateID, params)
    .then((res) => {
      document.getElementById("name").value = "";
      document.getElementById("email").value = "";
      document.getElementById("message").value = "";
      console.log(res);
      alert("your details sent successfully");
    })
    .catch((err) => console.log(err));
}

function validateEmail(email) {
    var re = /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/;
    return re.test(email);
}