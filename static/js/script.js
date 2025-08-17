
function displayMessage(text, type='info') {
    var toastEl = document.getElementById('messageToast');

    var toastBody = toastEl.querySelector('.toast-body');
    toastBody.textContent = text;

    toastEl.classList.remove('bg-primary', 'bg-success', 'bg-danger', 'bg-warning', 'bg-info', 'bg-secondary', 'bg-dark', 'bg-light');
    if (type == 'error') {
        toastEl.classList.add('bg-danger');
    } else if (type == 'warning') {
        toastEl.classList.add('bg-warning');
    } else if (type == 'success') {
        toastEl.classList.add('bg-success');
    } else if (type == 'info') {
        toastEl.classList.add('bg-info');
    } else if (type == 'secondary') {
        toastEl.classList.add('bg-secondary');
    } else if (type == 'dark') {
        toastEl.classList.add('bg-dark');
    } else if (type == 'light') {
        toastEl.classList.add('bg-light');
    } else if (type == 'primary') {
        toastEl.classList.add('bg-primary');
    }

    var toast = new bootstrap.Toast(toastEl);
    toast.show();
}

function showError(text) {
    displayMessage(text, 'error')
}

function showInfo(text) {
    displayMessage(text, 'error')
}

function showWarning(text) {
    displayMessage(text, 'error')
}

function showSuccess(text) {
    displayMessage(text, 'error')
}

function translateText() {
    const sourceLanguage = document.getElementById('sourceLang').value;
    const targetLanguage = document.getElementById('targetLang').value;

    if (sourceLanguage == targetLanguage) {
        showError('Source and Target Language can not be same.')
        return;
    }

    const text = document.getElementById('inputText').value;
    const requestBody = {
      "text": text,
      "source_language": sourceLanguage.toUpperCase(),
      "target_language": targetLanguage.toUpperCase()
    }

    showLoadingSpinner();
    fetch('/app/api/v1/translate', {
        method: 'POST',
        body: JSON.stringify(requestBody),
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data && data.detail) {
            errorMessage = ''
            data.detail.forEach(errorDetail => {
                errorMessage += `\n ${errorDetail.msg} \n`
            });
            showError(`Error translating .. ${errorMessage}`);
        } else {
            document.getElementById('translatedText').value = data;
        }
    }).catch(error => {
        showError('Error translating ..', error);
    }).finally(()=> {
        hideLoadingSpinner();
    });
}

function hideLoadingSpinner() {
  document.getElementById("spinnerBackdrop").style.display = "none";
}

function showLoadingSpinner() {
  document.getElementById("spinnerBackdrop").style.display = "flex";
}

//const textInputElement = document.getElementById('inputText');
//
//if (textInputElement) {
//    textInputElement.addEventListener('keydown', function (e) {
//        const triggerKeys = ['Enter'];
//
//        if (triggerKeys.includes(e.key)) {
//            setTimeout(translate, 50);
//        }
//    });
//}



