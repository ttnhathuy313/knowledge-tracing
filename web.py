import streamlit as st
from infer import query

st.title('Performance Predictor')
st.header('Description')
st.markdown('This app predicts the performance of students using SAKT model on ASSISTments2015 dataset.\
            \n For implementation details and technical report, please visit my [GitHub repo](https://github.com/ttnhathuy313/knowledge-tracing)')
st.header('Functionality')
st.markdown('There are 100 problem sets. You, as a student, can answer questions from the problem sets.\
            Based on your previous performance, SAKT guesses how likely you can answer the next question')
st.markdown('When you choose a problem set, you automatically answer the first unanswered question in that set. In addition, you can only answer it once.\
    ')
st.markdown('Because the dataset does not provide the actual content of questions, you have to decide whether you answer correctly or not')
st.markdown('To answer the question of the current problem set, either click "Answer correctly" or "Answer incorrectly"')
st.header('Demonstration')

if ('num_questions' not in st.session_state):
    st.session_state.num_questions = 0
if ('prev_id' not in st.session_state):
    st.session_state.prev_id = []
if ('prev_correct' not in st.session_state):
    st.session_state.prev_correct = []
    
st.markdown('Number of previously answered question: {}'.format(st.session_state.num_questions))
st.write('Previous problem sets: ',str(st.session_state.prev_id))
q_id = st.number_input('Pick the problem set id (from 1 to 100):', 1, 100)
st.write('The probability of you answering this question correctly: {}%'\
        .format(query(st.session_state.prev_id, st.session_state.prev_correct, q_id)))
if 'prev_q_id' in st.session_state and q_id != st.session_state.prev_q_id:
    st.write('You changed the problem set from {} to {}'.format(st.session_state.prev_q_id, q_id))
st.session_state.prev_q_id = q_id

if (st.button('Answer incorrectly')):
    st.session_state.prev_id.append(q_id)
    st.session_state.prev_correct.append(0)
    st.session_state.num_questions += 1
if (st.button('Answer correctly')):
    st.session_state.prev_id.append(q_id)
    st.session_state.prev_correct.append(1)
    st.session_state.num_questions += 1
    
if (st.session_state.num_questions == 120):
    st.session_state.num_questions = 0
    st.session_state.prev_id = []
    st.session_state.prev_correct = []
