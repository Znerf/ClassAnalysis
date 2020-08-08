

let app= new Vue({
    el:"#app",
    data:{
        maxdata:14,
        msg:"Hello World",
        base:"annotateData/output/data1/",
        img1:'annotateData/output/data1/sampada/1.jpg',
        img:'annotateData/output/data1/frame/1.jpg',
        counter:1,
        facenum:0,

        face:"SAMPADA",
        annotateData:[],
        inputarray:["sampada","sagar","pravesh","alex","sunil","unknown","Donot%20k"],
        
        

        data:['ATTENTIVE',"NON-ATTENTIVE","CONFUSED","SLEEP","UNKNOWN"],
        peoples:["SAMPADA","SAGAR","PRAVESH","ALEX","SUNIL","UNKNOWN"],
        
        alldata:[['num',"sampada","data","sagar","data","pravesh","data","data","alex","sunil","data","unknown","data","Donot%20k","data"]],
        alldatatemp:['1'],
        
    },methods:{
        checkForm: function () {
            this.update()
        },
        image_error(){
            console.log("Error")
            
            this.update()
        },
        complete(){
            this.download_csv();
            alert("complete")
            console.log("Thank you")
        },
        update(){
            this.alldatatemp.push(this.face)
            this.alldatatemp.push(this.annotateData)
            this.annotateData=""
            this.face=""

            this.facenum = this.facenum+1
            
            if (this.facenum % 7 ==0){
                this.facenum=0;
                this.counter=this.counter+1
                
                
                this.alldata.push(this.alldatatemp)
                console.log(this.alldata)

                this.alldatatemp=[]
                this.alldatatemp.push(this.counter)
                console.log(this.alldatatemp)

                if (this.counter==this.maxdata){
                    this.complete()
                }

                this.img= this.base+"frame/"+this.counter+".jpg"
            }
            this.face=this.peoples[this.facenum]
            this.img1= this.base+this.inputarray[this.facenum]+"/"+this.counter+".jpg"
        },

        download_csv() {
            var csv = 'Name,Title\n';
            this.alldata.forEach(function(row) {
                    csv += row.join(',');
                    csv += "\n";
            });
        
            console.log(csv);
            var hiddenElement = document.createElement('a');
            hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURI(csv);
            hiddenElement.target = '_blank';
            hiddenElement.download = 'people.csv';
            hiddenElement.click();
        }
      }


})
