// static/js/firebase_config.js
// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAB-Ws3omTYI5_31cg-gLojSZao93Ubrl8",
  authDomain: "hackathon-3dfc1.firebaseapp.com",
  projectId: "hackathon-3dfc1",
  storageBucket: "hackathon-3dfc1.firebasestorage.app",
  messagingSenderId: "141764911487",
  appId: "1:141764911487:web:e56dbf23043bfbd4c0fa4a",
  measurementId: "G-PRNL2HYSDH"
};

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const auth = firebase.auth();
