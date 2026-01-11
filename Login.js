import React, { useState } from 'react';
import './Login.css';
const Login = ({ SendLoginDetails, sendSignupRequest}) => {
    const [username,SetUsername] = useState('')
    const [password,SetPassword] = useState('')
    const [errorMessage, setErrorMessage] = useState('');
    const handleSubmit = async (event) => {
        event.preventDefault()
        console.log(username)
        console.log(password)
        const LoginSuccess =  await SendLoginDetails(username,password);
        if (LoginSuccess !== 'Login successful') {
            setErrorMessage(LoginSuccess);
        };


};

const handleSignup = async (event) => {
    console.log("Sign up requested")
    const SignupSuccess =  await sendSignupRequest(username,password);
    };

    return (
        <div className='login-container'>
            <div className = 'box'>
                <h1>Log-In</h1>
                <form onSubmit={handleSubmit}>
                    <div className='username'>
                        <input type="username" placeholder="Username"
                        onChange={e => SetUsername(e.target.value)}/>
                    </div>
                    <div className='password'>
                        <input type="password" placeholder="Password"
                        onChange={e => SetPassword(e.target.value)}/>
                    </div>
                    <button className='login' onClick={handleSubmit}>Log in</button>
                </form>
                <hr></hr>
                <button className='register' onClick={handleSignup}>Register Now</button>
                <div className='logo'></div>
                {errorMessage && (
                    <div className="error">
                        {errorMessage}
                    </div>
                )}
            </div>
        </div>
    );
};

export default Login;