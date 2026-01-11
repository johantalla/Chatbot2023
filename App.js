import React, { useState } from 'react';
import Login from './Login';
import Chat from './Chat';
import Signup from './Signup';

const App = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const SendSignupDetails = async(username,password) => {
    console.log("new username:", username)
    console.log("new password:", password)
    try {
      const response = await fetch('http://127.0.0.1:5000/signup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username,password }),
      });

    if (!response.ok) {
      if (response.status === 401) {
        return 'Username Taken';
      } else if (response.status === 403) {
        return 'Username may not contain special characters '
      } else if (response.status === 404) {
        return 'Username or password not entered'
      }
      console.error("Server responded with status:", response.status);
      return 'Sorry, there was an error with your request.';
    }
    const data = await response.json();
      if (data.success) {
        setIsLoggedIn(true);
      }
      console.log("Server response:",data)
      return data;
    } catch (error) {
      console.error('Error with request:', error);
      return 'Sorry, there was an error your request';
    }
  };
  const SendLoginDetails = async (username,password) => {
    console.log("Sending username:", username)
    console.log("Sending password:", password)
    try {
      const response = await fetch('http://127.0.0.1:5000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username,password }),
      });
      

      if (!response.ok) {
        if (response.status === 401) {
          return 'Username or Password Incorrect. Please try again';
        } else if (response.status === 404) {
          return 'User not found. Please try again.'
        }
        console.error("Server responded with status:", response.status);
        return 'Sorry, there was an error with your request.';
      }

      const data = await response.json();
      if (data.success) {
        setIsLoggedIn(true);
      }
      console.log("Server response:",data)
      return data;
    } catch (error) {
      console.error('Error with request:', error);
      return 'Sorry, there was an error your request';
    }
  };

  const sendSignUpRequest = () => {
    setIsLoggedIn('signup')
  }

  const sendLogoutRequest = () => {
    setIsLoggedIn(false)
  }

  const sendMessageToBot = async (message) => {
    console.log(message)
    try {
      const response = await fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });

      const data = await response.json();
      return data.response;
    } catch (error) {
      console.error('Error communicating with the bot:', error);
      return 'Sorry, there was an error communicating with the bot.';
    }
  };

  return (
    <div>
      {isLoggedIn === true ? (
        <Chat sendMessageToBot={sendMessageToBot} sendLogoutRequest = {sendLogoutRequest}  />
      ) : isLoggedIn === false ? (
        <Login SendLoginDetails={SendLoginDetails} sendSignupRequest = {sendSignUpRequest} />
      ): isLoggedIn=== "signup" ? (
        <Signup sendSignupDetails={SendSignupDetails}/>
      ):null}
    </div>
  );
};

export default App;