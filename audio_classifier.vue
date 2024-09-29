<template>
  <div>
    <button class="classify-button" @click="classifyAudio">Classify Audio</button>
  </div>
</template>

<style scoped>
.classify-button {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  background-color: blue;
  color: grey;
  border: 2px solid white;
  font-size: 16px;
  box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
  transition: transform 0.2s, box-shadow 0.2s;
}

.classify-button:active {
  transform: translateY(2px);
  box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
}
</style>
