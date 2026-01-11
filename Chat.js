import React, { useEffect, useState,  useRef } from 'react';
import './Chat.css'; 

const Chat = ({ sendMessageToBot, sendLogoutRequest }) => {
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [dropdownVisible, setDropdownVisible] = useState(false);
  const [isBrightMode, setIsBrightMode] = useState(false);
  const messagesEndRef = useRef(null);


  useEffect(() => {
      if (messagesEndRef.current) {
        console.log('Scrolling to bottom...');
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
      } else {
        console.log('messagesEndRef is null');
      }
  }, [messages]);


  const handleSendMessage = async () => {
    if (newMessage.trim()) {
      const timestamp = new Date().toLocaleTimeString();
    const updatedMessages = [...messages, { text: newMessage, sender: 'user', time: timestamp }];
    setMessages(updatedMessages);
    const botResponse = await sendMessageToBot(newMessage);
    setMessages([...updatedMessages, { text: botResponse, sender: 'bot', time: timestamp }]);
    setNewMessage('');
    }
  };

  const handleLogout = async (event) => {
    console.log("Log out requested")
    const LogoutSuccess =  await sendLogoutRequest();
    };

  const toggleDropdown = () => {
    setDropdownVisible(!dropdownVisible);
  };

  const handleThemeToggle = () => {
    console.log("theme changed")
    setIsBrightMode(!isBrightMode); /*Theme toggle */
    console.log(isBrightMode)
  };

  return (
    <div className="screen-container">
      <div className="sidebar">
        <div className='side-text'>
          <p>HELOOOOOOO1OOOOOOOOOOOOOOOOOOO1OOOO</p>
          <p>BYWEBYESEEYOULATERALLIGATOR</p>
        </div>
      </div>
      <div className='right-container'>
        <div className='top-bar'>
          <div className='top-bar-text'>
            <p>Chatterbox</p>
          </div>
          <div className="user-icon">
            <button onClick={toggleDropdown}>
              <div className="user-icon-btn"></div> {/*icon*/}
            </button>
            {dropdownVisible && (
              <div className="dropdown-menu">
                <label class="switch">
                <input 
                  type="checkbox"
                  checked={isBrightMode}
                  onChange={handleThemeToggle} 
                />
                <span class="slider round"></span>
              </label>
                <button onClick={handleLogout}>Logout</button>
              </div>
              )}
          </div>
        </div>
        <div className='chat-container'>
          <div className="messages-container ${isBrightMode ? 'dark-mode' : 'bright-mode'}">
          {console.log("Current theme:", isBrightMode ? "dark-mode" : "bright-mode")}
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.sender}`}>
              <span className="message-text">{message.text}</span>
              <div className="message-time">{message.time}</div>
            </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
          <div className="input-container">
            <input
              type="text"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              placeholder="Type a message..."
              onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <button onClick={handleSendMessage}>Send</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;
