import React, { useState } from 'react';
import './Signup.css';
const Signup = ({ sendSignupDetails }) => {
    const [username,SetUsername] = useState('')
    const [password,SetPassword] = useState('')
    const [errorMessage, setErrorMessage] = useState('');
    const handleSubmit = async (event) => {
        event.preventDefault()
        console.log(username)
        console.log(password)
        const SignupSuccess =  await sendSignupDetails(username,password);
        if (SignupSuccess !== 'Login successful') {
            setErrorMessage(SignupSuccess);
        };
}

    return (
        <div className='signup-container'>
            <div className = 'box'>
                <h1>Sign-Up</h1>
                <form onSubmit={handleSubmit}>
                    <div className='username'>
                        <input type="username" placeholder="Username"
                        onChange={e => SetUsername(e.target.value)}/>
                    </div>
                    <div className='password'>
                        <input type="password" placeholder="Password"
                        onChange={e => SetPassword(e.target.value)}/>
                    </div>
                    <button className='signup' onClick={handleSubmit}>Sign Up</button>
                    <hr></hr>
                    <div className='logo'></div>
                </form>
                {errorMessage && (
                    <div className="error">
                        {errorMessage}
                    </div>
                )}
            </div>
        </div>
    )
};

export default Signup;